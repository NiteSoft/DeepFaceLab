from functools import partial

import cv2
import numpy as np

from facelib import FaceType
from interact import interact as io
from mathlib import get_power_of_two
from models import ModelBase
from nnlib import nnlib
from samplelib import *

from facelib import PoseEstimator

class CONVModel(ModelBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                            ask_sort_by_yaw=False,
                            ask_random_flip=False,
                            ask_src_scale_mod=False)

    #override
    def onInitializeOptions(self, is_first_run, ask_override):
        def_resolution = 128
        if is_first_run:
            self.options['resolution'] = io.input_int("Resolution ( 128,256 ?:help skip:%d) : " % def_resolution, def_resolution, [128,256], help_message="More resolution requires more VRAM and time to train. Value will be adjusted to multiple of 16.")
        else:
            self.options['resolution'] = self.options.get('resolution', def_resolution)


    #override
    def onInitialize(self, batch_size=-1, **in_options):
        exec(nnlib.code_import_all, locals(), globals())
        self.set_vram_batch_requirements({2:1})
        CONVModel.initialize_nn_functions()
        
        resolution = self.resolution = self.options['resolution']
        bgr_shape = (resolution, resolution, 3)
        bgr_t_shape = (resolution, resolution, 6)
        mask_shape = (resolution, resolution, 1)

        self.C = modelify(CONVModel.UNet ( 3, use_batch_norm=True, ngf=64))(Input(bgr_t_shape))
        self.D = modelify(CONVModel.Discriminator(ndf=256) ) (Input(bgr_shape))   
        
        if not self.is_first_run():
            self.load_weights_safe( self.get_model_filename_list()  )            
        else:
            conv_weights_list = []
            for model in [self.C, self.D]:
                for layer in model.layers:
                    if type(layer) == keras.layers.Conv2D:
                        conv_weights_list += [layer.weights[0]] #Conv2D kernel_weights
            #CAInitializerMP ( conv_weights_list )
            
        warped_A = Input(bgr_shape)
        warped_Am = Input(mask_shape)
        real_A = Input(bgr_shape)
        real_Am = Input(mask_shape)
        
        real_B = Input(bgr_shape)
        real_Bm = Input(mask_shape)
        
        C_real_A = self.C ( K.concatenate ( [ ( (warped_A+1)*warped_Am + (1-warped_Am) ) -1,
                                              ( (real_A+1)*(1-real_Am) + real_Am) -1,
                                            ] , axis=-1) )
                                            
        C_real_B = self.C ( K.concatenate ( [( (real_B+1)*real_Bm + (1-real_Bm) ) -1,
                                             ( (real_A+1)*(1-real_Am) + real_Am) -1,
                                            ] , axis=-1) )
                                                
        real_A_d = self.D( real_A )
        real_A_d_ones = K.ones_like(real_A_d)    
            
        C_real_B_d = self.D( C_real_B )
        C_real_B_d_ones = K.ones_like(C_real_B_d)
        C_real_B_d_zeros = K.zeros_like(C_real_B_d)

        if self.is_training_mode:
            def opt(lr=2e-5):
                return Adam(lr=lr, beta_1=0.5, beta_2=0.999, tf_cpu_mode=2)#, clipnorm=1)
                
            def DLoss(labels,logits):
                return K.mean(K.binary_crossentropy(labels,logits))
                
            self.G_view = K.function([warped_A, warped_Am, real_A, real_Am, real_B, real_Bm], [C_real_A, C_real_B])
    
            #K.mean( 10 * dssim(kernel_size=int(resolution/11.6),max_value=1.0) ( C_real_A, real_A ) ) + \
            loss_C = 50 * K.mean( K.square ( C_real_A - real_A ) ) + \
                     DLoss(C_real_B_d_ones, C_real_B_d )
                 
            weights_C = self.C.trainable_weights
            
            self.C_train = K.function ([ warped_A, warped_Am, real_A, real_Am, real_B, real_Bm],[loss_C], opt().get_updates(loss_C, weights_C) )
            
           
            
            loss_D = (DLoss(real_A_d_ones, real_A_d ) + \
                      DLoss(C_real_B_d_zeros, C_real_B_d ) ) * 0.5
                      
            self.D_train = K.function ([ warped_A, warped_Am, real_A, real_Am, real_B, real_Bm],[loss_D], opt().get_updates(loss_D, self.D.trainable_weights) )
                
            ###########
            t = SampleProcessor.Types
            generators = [
                                           
                    SampleGeneratorFace(self.training_data_src_path, debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(random_flip=True),
                        output_sample_types=[ {'types': (t.IMG_TRANSFORMED, t.FACE_TYPE_FULL, t.MODE_BGR_SHUFFLE), 'resolution':self.resolution, 'normalize_tanh':True},
                                              {'types': (t.IMG_TRANSFORMED, t.FACE_TYPE_FULL, t.MODE_M), 'resolution':self.resolution},
                                              {'types': (t.IMG_SOURCE, t.FACE_TYPE_FULL, t.MODE_BGR), 'resolution':self.resolution, 'normalize_tanh':True},
                                              {'types': (t.IMG_SOURCE, t.FACE_TYPE_FULL, t.MODE_M), 'resolution':self.resolution}
                                              ] ),

                    SampleGeneratorFace(self.training_data_src_path, debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(random_flip=True),
                        output_sample_types=[ {'types': (t.IMG_SOURCE, t.FACE_TYPE_FULL, t.MODE_BGR), 'resolution':self.resolution, 'normalize_tanh':True},
                                              {'types': (t.IMG_SOURCE, t.FACE_TYPE_FULL, t.MODE_M), 'resolution':self.resolution}
                                            ] ),                                             
                   ]
                   
            self.set_training_data_generators (generators)
        else:
            self.G_convert = K.function([warped_B064],[rec_C_A0_B0])

    def get_model_filename_list(self):
        return [
                [self.C, 'C.h5'],
                [self.D, 'D.h5']
              ]
               
    #override
    def onSave(self):
        self.save_weights_safe( self.get_model_filename_list() )

    #override
    def onTrainOneIter(self, generators_samples, generators_list):
        warped_A, warped_Am, real_A, real_Am = generators_samples[0]
        real_B, real_Bm = generators_samples[1]
        loss_C, = self.C_train ( [warped_A, warped_Am, real_A, real_Am, real_B, real_Bm] )
        loss_D, = self.D_train ( [warped_A, warped_Am, real_A, real_Am, real_B, real_Bm] )
        
        return ( ('C', loss_C), ('D', loss_D) )

    #override
    def onGetPreview(self, sample):
        warped_A  = sample[0][0][0:4]
        warped_Am = sample[0][1][0:4]
        real_A    = sample[0][2][0:4]
        real_Am   = sample[0][3][0:4]
        
        real_B    = sample[1][0][0:4]
        real_Bm   = sample[1][1][0:4]

        G_view_result = self.G_view([warped_A, warped_Am, real_A, real_Am, real_B, real_Bm])

        warped_A, warped_Am, real_A, real_Am, real_B, real_Bm, C_real_A, C_real_B = \
          [ x[0]/2+0.5 for x in ( [warped_A, warped_Am, real_A, real_Am, real_B, real_Bm] + G_view_result )  ]

        #r = sample64x4
        r = np.concatenate ( (real_A, C_real_A, real_B, C_real_B), axis=1 )
        
        return [ ('CONVModel', r ) ]

    def predictor_func (self, inp_f0, inp_f1, inp_f2):        
        feed = [ inp_f0[np.newaxis,...], inp_f1[np.newaxis,...], inp_f2[np.newaxis,...] ]
        x = self.G_convert (feed)[0]
        return np.clip ( x[0], 0, 1)

    # #override
    # def get_converter(self, **in_options):
    #     from models import ConverterImage
    #     return ConverterImage(self.predictor_func,
    #                           predictor_input_size=self.options['resolution'],
    #                           **in_options)
    #override
    def get_converter(self):
        base_erode_mask_modifier = 30
        base_blur_mask_modifier = 0

        default_erode_mask_modifier = 0
        default_blur_mask_modifier = 0

        face_type = FaceType.FULL

        from converters import ConverterAvatar
        return ConverterAvatar(self.predictor_func,
                               predictor_input_size=64)


    @staticmethod
    def PatchDiscriminator(ndf=64, use_batch_norm=True):
        exec (nnlib.import_all(), locals(), globals())

        if not use_batch_norm:
            use_bias = True
            def XNormalization(x):
                return InstanceNormalization (axis=-1)(x)
        else:
            use_bias = False
            def XNormalization(x):
                return BatchNormalization (axis=-1)(x)

        XConv2D = partial(Conv2D, use_bias=use_bias)

        """
        def func(input):
            b,h,w,c = K.int_shape(input)

            x = input

            x = ZeroPadding2D((1,1))(x)
            x = XConv2D( ndf, 4, strides=2, padding='valid', use_bias=True)(x)
            x = LeakyReLU(0.2)(x)

            x = ZeroPadding2D((1,1))(x)
            x = XConv2D( ndf*2, 4, strides=2, padding='valid')(x)
            x = XNormalization(x)
            x = LeakyReLU(0.2)(x)

            x = ZeroPadding2D((1,1))(x)
            x = XConv2D( ndf*4, 4, strides=2, padding='valid')(x)
            x = XNormalization(x)
            x = LeakyReLU(0.2)(x)

            x = ZeroPadding2D((1,1))(x)
            x = XConv2D( ndf*8, 4, strides=2, padding='valid')(x)
            x = XNormalization(x)
            x = LeakyReLU(0.2)(x)

            x = ZeroPadding2D((1,1))(x)
            x = XConv2D( ndf*8, 4, strides=2, padding='valid')(x)
            x = XNormalization(x)
            x = LeakyReLU(0.2)(x)

            x = ZeroPadding2D((1,1))(x)
            return XConv2D( 1, 4, strides=1, padding='valid', use_bias=True, activation='sigmoid')(x)#
        """
        def func(input):
            b,h,w,c = K.int_shape(input)

            x = input

            x = XConv2D( ndf, 4, strides=2, padding='same')(x)
            x = LeakyReLU(0.2)(x)
            x = XNormalization(x)
            
            x = XConv2D( ndf*2, 4, strides=2, padding='same')(x)
            x = LeakyReLU(0.2)(x)
            x = XNormalization(x)
            
            x = XConv2D( ndf*4, 4, strides=2, padding='same')(x)
            x = LeakyReLU(0.2)(x)
            x = XNormalization(x)
            
            x = XConv2D( ndf*8, 4, strides=2, padding='same')(x)
            x = LeakyReLU(0.2)(x)
            x = XNormalization(x)
            
            x = XConv2D( ndf*16, 4, strides=2, padding='same')(x)
            x = LeakyReLU(0.2)(x)
            x = XNormalization(x)

            return XConv2D( 1, 5, strides=1, padding='same', use_bias=True, activation='sigmoid')(x)#
        return func
   
    @staticmethod
    def EncFlow(padding='zero', **kwargs):
        exec (nnlib.import_all(), locals(), globals())

        use_bias = False
        def XNorm(x):
            return BatchNormalization (axis=-1)(x)
        XConv2D = partial(Conv2D, padding=padding, use_bias=use_bias)

        def downscale (dim):
            def func(x):
                return LeakyReLU(0.1)( Conv2D(dim, 5, strides=2, padding='same')(x))
            return func

        def upscale (dim):
            def func(x):
                return SubpixelUpscaler()(LeakyReLU(0.1)(Conv2D(dim * 4, 3, strides=1, padding='same')(x)))
            return func

               
        def func(input):
            x, = input
            b,h,w,c = K.int_shape(x)
            x = downscale(64)(x)
            x = downscale(128)(x)
            x = downscale(256)(x)
            x = downscale(512)(x)

            x = Dense(512)(Flatten()(x))
            x = Dense(4 * 4 * 512)(x)
            x = Reshape((4, 4, 512))(x) 
            x = upscale(512)(x)   
            return x
            
        return func

    @staticmethod
    def Dec64Flow(output_nc=3, **kwargs):
        exec (nnlib.import_all(), locals(), globals())

        ResidualBlock = CONVModel.ResidualBlock
        upscale = CONVModel.upscale
        to_bgr = CONVModel.to_bgr

        def func(input):
            x = input[0]
            
            x = upscale(512)(x)
            x = upscale(256)(x)
            x = upscale(128)(x)
            return to_bgr(output_nc, activation="tanh") (x)

        return func
        
    @staticmethod
    def Discriminator(ndf=128):
        exec (nnlib.import_all(), locals(), globals())

        #use_bias = True
        #def XNormalization(x):
        #    return InstanceNormalization (axis=-1)(x)
        use_bias = False
        def XNormalization(x):
            return BatchNormalization (axis=-1)(x)

        XConv2D = partial(Conv2D, use_bias=use_bias)

        def func(input):
            b,h,w,c = K.int_shape(input)

            x = input

            x = XConv2D( ndf, 3, strides=2, padding='same', use_bias=True)(x)
            x = LeakyReLU(0.2)(x)

            x = XConv2D( ndf*2, 3, strides=2, padding='same')(x)
            x = XNormalization(x)
            x = LeakyReLU(0.2)(x)
            
            x = XConv2D( ndf*4, 3, strides=2, padding='same')(x)
            x = XNormalization(x)
            x = LeakyReLU(0.2)(x)
            
            
            return XConv2D( 1, 4, strides=1, padding='same', use_bias=True, activation='sigmoid')(x)#
        return func
        
    @staticmethod
    def UNet(output_nc, use_batch_norm, ngf=64):
        exec (nnlib.import_all(), locals(), globals())

        if not use_batch_norm:
            use_bias = True
            def XNormalizationL():
                return InstanceNormalization (axis=-1)
        else:
            use_bias = False
            def XNormalizationL():
                return BatchNormalization (axis=-1)
                
        def XNormalization(x):
            return XNormalizationL()(x)
                
        XConv2D = partial(Conv2D, padding='same', use_bias=use_bias)
        XConv2DTranspose = partial(Conv2DTranspose, padding='same', use_bias=use_bias)
      
        def func(input):
            
            b,h,w,c = K.int_shape(input)
            
            n_downs = get_power_of_two(w) - 4
            
            Norm = XNormalizationL()
            Norm2 = XNormalizationL()
            Norm4 = XNormalizationL()
            Norm8 = XNormalizationL()
            
            x = input
            
            x = e1 = XConv2D( ngf, 4, strides=2, use_bias=True ) (x)

            x = e2 = Norm2( XConv2D( ngf*2, 4, strides=2  )( LeakyReLU(0.2)(x) ) )
            x = e3 = Norm4( XConv2D( ngf*4, 4, strides=2  )( LeakyReLU(0.2)(x) ) )
            
            l = []
            for i in range(n_downs):
                x = Norm8( XConv2D( ngf*8, 4, strides=2  )( LeakyReLU(0.2)(x) ) )
                l += [x]
            
            x = XConv2D( ngf*8, 4, strides=2, use_bias=True  )( LeakyReLU(0.2)(x) )
            
            for i in range(n_downs):
                x = Norm8( XConv2DTranspose( ngf*8, 4, strides=2  )( ReLU()(x) ) )
                if i <= n_downs-2:
                    x = Dropout(0.5)(x)                
                x = Concatenate(axis=-1)([x, l[-i-1] ])
  
            x = Norm4( XConv2DTranspose( ngf*4, 4, strides=2  )( ReLU()(x) ) )
            x = Concatenate(axis=-1)([x, e3])

            x = Norm2( XConv2DTranspose( ngf*2, 4, strides=2  )( ReLU()(x) ) )
            x = Concatenate(axis=-1)([x, e2])  
            
            x = Norm( XConv2DTranspose( ngf, 4, strides=2  )( ReLU()(x) ) )
            x = Concatenate(axis=-1)([x, e1])   
            
            x = XConv2DTranspose(output_nc, 4, strides=2, activation='tanh', use_bias=True)( ReLU()(x) )

            return x
        return func
            
    @staticmethod
    def ResNet(output_nc, use_batch_norm, ngf=64, n_blocks=6, use_dropout=False):
        exec (nnlib.import_all(), locals(), globals())

        if not use_batch_norm:
            use_bias = True
            def XNormalization(x):
                return InstanceNormalization (axis=-1)(x)
        else:
            use_bias = False
            def XNormalization(x):
                return BatchNormalization (axis=-1)(x)
                
        XConv2D = partial(Conv2D, padding='same', use_bias=use_bias)
        XConv2DTranspose = partial(Conv2DTranspose, padding='same', use_bias=use_bias)

        def func(input):


            def ResnetBlock(dim, use_dropout=False):
                def func(input):
                    x = input

                    x = XConv2D(dim, 3, strides=1)(x)
                    x = XNormalization(x)
                    x = ReLU()(x)

                    if use_dropout:
                        x = Dropout(0.5)(x)

                    x = XConv2D(dim, 3, strides=1)(x)
                    x = XNormalization(x)
                    x = ReLU()(x)
                    return Add()([x,input])
                return func

            x = input

            x = ReLU()(XNormalization(XConv2D(ngf, 7, strides=1)(x)))

            x = ReLU()(XNormalization(XConv2D(ngf*2, 3, strides=2)(x)))
            x = ReLU()(XNormalization(XConv2D(ngf*4, 3, strides=2)(x)))

            for i in range(n_blocks):
                x = ResnetBlock(ngf*4, use_dropout=use_dropout)(x)

            x = ReLU()(XNormalization(XConv2DTranspose(ngf*2, 3, strides=2)(x)))
            x = ReLU()(XNormalization(XConv2DTranspose(ngf  , 3, strides=2)(x)))

            x = XConv2D(output_nc, 7, strides=1, activation='sigmoid', use_bias=True)(x)

            return x

        return func
        
    @staticmethod
    def initialize_nn_functions():
        exec (nnlib.import_all(), locals(), globals())

        class ResidualBlock(object):
            def __init__(self, filters, kernel_size=3, padding='zero', **kwargs):
                self.filters = filters
                self.kernel_size = kernel_size
                self.padding = padding

            def __call__(self, inp):
                x = inp
                x = Conv2D(self.filters, kernel_size=self.kernel_size, padding=self.padding)(x)
                x = LeakyReLU(0.2)(x)
                x = Conv2D(self.filters, kernel_size=self.kernel_size, padding=self.padding)(x)
                x = Add()([x, inp])
                x = LeakyReLU(0.2)(x)
                return x
        CONVModel.ResidualBlock = ResidualBlock

        def downscale (dim, padding='zero', act='', **kwargs):
            def func(x):
                return LeakyReLU(0.2) (Conv2D(dim, kernel_size=5, strides=2, padding=padding)(x))
            return func
        CONVModel.downscale = downscale

        def upscale (dim, padding='zero', norm='', act='', **kwargs):
            def func(x):
                return SubpixelUpscaler()( LeakyReLU(0.2)(Conv2D(dim * 4, kernel_size=3, strides=1, padding=padding)(x)))
            return func
        CONVModel.upscale = upscale

        def to_bgr (output_nc, padding='zero', activation='sigmoid', **kwargs):
            def func(x):
                return Conv2D(output_nc, kernel_size=5, padding=padding, activation=activation)(x)
            return func
        CONVModel.to_bgr = to_bgr
              
        
Model = CONVModel

""" 
def BCELoss(logits, ones):
    if ones:
        return K.mean(K.binary_crossentropy(K.ones_like(logits),logits))
    else:
        return K.mean(K.binary_crossentropy(K.zeros_like(logits),logits))

def MSELoss(labels,logits):
    return K.mean(K.square(labels-logits))

def DLoss(labels,logits):
    return K.mean(K.binary_crossentropy(labels,logits))

def MAELoss(t1,t2):
    return dssim(kernel_size=int(resolution/11.6),max_value=2.0)(t1+1,t2+1 )
    return K.mean(K.abs(t1 - t2) )
"""