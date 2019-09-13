from functools import partial
import numpy as np

from nnlib import nnlib
from models import ModelBase
from facelib import FaceType
from samplelib import *
from interact import interact as io
import cv2

#SAE - Styled AutoEncoder
class SAEModel(ModelBase):

    encoderH5 = 'encoder.h5'
    inter_BH5 = 'inter_B.h5'
    inter_ABH5 = 'inter_AB.h5'
    decoderH5 = 'decoder.h5'
    decodermH5 = 'decoderm.h5'

    decoder_srcH5 = 'decoder_src.h5'
    decoder_srcmH5 = 'decoder_srcm.h5'
    decoder_dstH5 = 'decoder_dst.h5'
    decoder_dstmH5 = 'decoder_dstm.h5'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                            ask_batch_size=False)

    #override
    def onInitializeOptions(self, is_first_run, ask_override):
        yn_str = {True:'y',False:'n'}


        default_archi = 'df'
        default_face_type = 'f'
        default_learn_mask = False
        if is_first_run:
            self.options['face_type'] = io.input_str ("Half or Full face? (h/f, ?:help skip:f) : ", default_face_type, ['h','f'], help_message="Half face has better resolution, but covers less area of cheeks.").lower()
            self.options['learn_mask'] = io.input_bool ("Learn mask? (y/n, ?:help skip:y) : ", default_learn_mask, help_message="Learning mask can help model to recognize face directions. Learn without mask can reduce model size, in this case converter forced to use 'not predicted mask' that is not smooth as predicted. Model with style values can be learned without mask and produce same quality result.")
        else:
            self.options['face_type'] = self.options.get('face_type', default_face_type)
            self.options['learn_mask'] = self.options.get('learn_mask', default_learn_mask)

        self.upscale_count = stage_max = 3
        
        default_stage = self.options.get('stage', 0)
        default_target_stage_iter = self.options.get('target_stage_iter', self.iter+30000 )

        if (is_first_run or ask_override):
            new_stage = np.clip ( io.input_int("Stage ( 0-%d skip:%d) : " % (stage_max-1,default_stage), default_stage), 0, stage_max-1 )
            if new_stage != default_stage:
                self.options['start_stage_iter'] = self.iter
                default_target_stage_iter = self.iter+30000
            self.options['stage'] = new_stage
        else:
            self.options['stage'] = self.options.get('stage', default_stage)

        if self.options['stage'] == 0:
            if 'start_stage_iter' in self.options:
                self.options.pop('start_stage_iter')

            if 'target_stage_iter' in self.options:
                self.options.pop('target_stage_iter')
        else:
            if (is_first_run or ask_override):
                self.options['target_stage_iter'] = io.input_int("Target stage iteration (skip:%d) : " % default_target_stage_iter, default_target_stage_iter)
            else:
                self.options['target_stage_iter'] = self.options.get('target_stage_iter', default_target_stage_iter)

        default_batch_size = {0:20, 1:8, 2:4}.get (self.options['stage'], 4)
        default_batch_size = self.options.get('batch_size', default_batch_size)
        if (is_first_run or ask_override):
            self.batch_size = self.options['batch_size'] = max(0, io.input_int("Batch_size (?:help skip:%d) : " % (default_batch_size), default_batch_size, help_message="Larger batch size is better for NN's generalization, but it can cause Out of Memory error. Tune this value for your videocard manually."))
        else:
            self.batch_size = self.options['batch_size'] = self.options.get('batch_size', default_batch_size)


        def_optimizer_mode = self.options.get('optimizer_mode', 3)
        if (is_first_run or ask_override) and 'tensorflow' in self.device_config.backend:
            self.options['optimizer_mode'] = io.input_int ("Optimizer mode? ( 1,2,3 ?:help skip:%d) : " % (def_optimizer_mode), def_optimizer_mode, help_message="1 - no changes. 2 - allows you to train x2 bigger network consuming RAM. 3 - allows you to train x3 bigger network consuming huge amount of RAM and slower, depends on CPU power.")
        else:
            self.options['optimizer_mode'] = self.options.get('optimizer_mode', def_optimizer_mode)

        #if is_first_run:
        #    self.options['archi'] = io.input_str ("AE architecture (df, liae ?:help skip:%s) : " % (default_archi) , default_archi, ['df','liae'], help_message="'df' keeps faces more natural. 'liae' can fix overly different face shapes.").lower() #-s version is slower, but has decreased change to collapse.
        #else:
        #    self.options['archi'] = self.options.get('archi', default_archi)
        self.options['archi'] = 'df'

        default_ae_dims = 256# if 'liae' in self.options['archi'] else 256
        default_e_dims = 64
        default_d_dims = 256
        def_ca_weights = False

        if is_first_run:
            self.options['ae_dims'] = np.clip ( io.input_int("AutoEncoder dims (32-1024 ?:help skip:%d) : " % (default_ae_dims) , default_ae_dims, help_message="All face information will packed to AE dims. If amount of AE dims are not enough, then for example closed eyes will not be recognized. More dims are better, but require more VRAM. You can fine-tune model size to fit your GPU." ), 32, 1024 )
            self.options['e_dims'] = np.clip ( io.input_int("Encoder dims ( ?:help skip:%d) : " % (default_e_dims) , default_e_dims, help_message="More encoder dims help to recognize more facial features, but require more VRAM. You can fine-tune model size to fit your GPU." ), 64, 512 )
            self.options['d_dims'] = np.clip ( io.input_int("Decoder dims ( ?:help skip:%d) : " % (default_d_dims) , default_d_dims, help_message="More decoder dims help to get better details, but require more VRAM. You can fine-tune model size to fit your GPU." ), 64, 1024 )
            self.options['ca_weights'] = io.input_bool ("Use CA weights? (y/n, ?:help skip: %s ) : " % (yn_str[def_ca_weights]), def_ca_weights, help_message="Initialize network with 'Convolution Aware' weights. This may help to achieve a higher accuracy model, but consumes a time at first run.")
        else:
            self.options['ae_dims'] = self.options.get('ae_dims', default_ae_dims)
            self.options['e_dims'] = self.options.get('e_dims', default_e_dims)
            self.options['d_dims'] = self.options.get('d_dims', default_d_dims)
            self.options['ca_weights'] = self.options.get('ca_weights', def_ca_weights)

        default_face_style_power = 0.0
        default_bg_style_power = 0.0
        if is_first_run or ask_override:
            def_pixel_loss = self.options.get('pixel_loss', False)
            self.options['pixel_loss'] = io.input_bool ("Use pixel loss? (y/n, ?:help skip: %s ) : " % (yn_str[def_pixel_loss]), def_pixel_loss, help_message="Pixel loss may help to enhance fine details and stabilize face color. Use it only if quality does not improve over time. Enabling this option too early increases the chance of model collapse.")

            default_face_style_power = default_face_style_power if is_first_run else self.options.get('face_style_power', default_face_style_power)
            self.options['face_style_power'] = np.clip ( io.input_number("Face style power ( 0.0 .. 100.0 ?:help skip:%.2f) : " % (default_face_style_power), default_face_style_power,
                                                                               help_message="Learn to transfer face style details such as light and color conditions. Warning: Enable it only after 10k iters, when predicted face is clear enough to start learn style. Start from 0.1 value and check history changes. Enabling this option increases the chance of model collapse."), 0.0, 100.0 )

            default_bg_style_power = default_bg_style_power if is_first_run else self.options.get('bg_style_power', default_bg_style_power)
            self.options['bg_style_power'] = np.clip ( io.input_number("Background style power ( 0.0 .. 100.0 ?:help skip:%.2f) : " % (default_bg_style_power), default_bg_style_power,
                                                                               help_message="Learn to transfer image around face. This can make face more like dst. Enabling this option increases the chance of model collapse."), 0.0, 100.0 )

            default_apply_random_ct = False if is_first_run else self.options.get('apply_random_ct', False)
            self.options['apply_random_ct'] = io.input_bool ("Apply random color transfer to src faceset? (y/n, ?:help skip:%s) : " % (yn_str[default_apply_random_ct]), default_apply_random_ct, help_message="Increase variativity of src samples by apply LCT color transfer from random dst samples. It is like 'face_style' learning, but more precise color transfer and without risk of model collapse, also it does not require additional GPU resources, but the training time may be longer, due to the src faceset is becoming more diverse.")

            if nnlib.device.backend != 'plaidML': # todo https://github.com/plaidml/plaidml/issues/301
                default_clipgrad = False if is_first_run else self.options.get('clipgrad', False)
                self.options['clipgrad'] = io.input_bool ("Enable gradient clipping? (y/n, ?:help skip:%s) : " % (yn_str[default_clipgrad]), default_clipgrad, help_message="Gradient clipping reduces chance of model collapse, sacrificing speed of training.")
            else:
                self.options['clipgrad'] = False

        else:
            self.options['pixel_loss'] = self.options.get('pixel_loss', False)
            self.options['face_style_power'] = self.options.get('face_style_power', default_face_style_power)
            self.options['bg_style_power'] = self.options.get('bg_style_power', default_bg_style_power)
            self.options['apply_random_ct'] = self.options.get('apply_random_ct', False)
            self.options['clipgrad'] = self.options.get('clipgrad', False)

        if is_first_run:
            self.options['pretrain'] = io.input_bool ("Pretrain the model? (y/n, ?:help skip:n) : ", False, help_message="Pretrain the model with large amount of various faces. This technique may help to train the fake with overly different face shapes and light conditions of src/dst data. Face will be look more like a morphed. To reduce the morph effect, some model files will be initialized but not be updated after pretrain: LIAE: inter_AB.h5 DF: encoder.h5. The longer you pretrain the model the more morphed face will look. After that, save and run the training again.")
        else:
            self.options['pretrain'] = False

    #override
    def onInitialize(self):
        exec(nnlib.import_all(), locals(), globals())

        ae_dims = self.options['ae_dims']
        e_dims = self.options['e_dims']
        d_dims = self.options['d_dims']
        learn_mask=self.options['learn_mask']
        self.pretrain = self.options['pretrain'] = self.options.get('pretrain', False)
        if not self.pretrain:
            self.options.pop('pretrain')

        

        self.stage = stage = self.options['stage']
        self.start_stage_iter = self.options.get('start_stage_iter', 0)
        self.target_stage_iter = self.options.get('target_stage_iter', 0)

        self.resolution = resolution = 256

        self.stage_resolution = stage_resolution = 32 * (2**stage)
        
        pixel_loss = self.options['pixel_loss']
        if self.stage_resolution < 64:
            pixel_loss = True
            
        bgr_shape = (resolution, resolution, 3)
        mask_shape = (resolution, resolution, 1)

        stage_bgr_shape  = ( stage_resolution, stage_resolution, 3)
        stage_mask_shape = ( stage_resolution, stage_resolution, 1)

        apply_random_ct = self.options.get('apply_random_ct', False)
        masked_training = True

        class WeightedSum(KL.Layer):
            def __init__(self, **kwargs):
                super(WeightedSum, self).__init__(**kwargs)
                self.alpha = K.variable( 0.0, name='ws_alpha')

            def build(self, input_shape):
                self.input_spec = [keras.engine.InputSpec(shape=x) for x in input_shape]

            def call(self, inputs):
                return (1.0 - self.alpha)*inputs[0] + self.alpha*inputs[1]

            def set_alpha(self, alpha):
                K.set_value(self.alpha, alpha)

        class SAEDFModel(object):
            def __init__(self, resolution, ae_dims, e_dims, d_dims, learn_mask, upscale_count=3):
                super().__init__()
                self.resolution = resolution
                self.ae_dims = ae_dims
                self.e_dims = e_dims
                self.d_dims = d_dims
                self.learn_mask = learn_mask
                self.upscale_count = upscale_count

                bgr_shape = (resolution, resolution, 3)
                mask_shape = (resolution, resolution, 1)

                def ResidualBlock(dim):
                    def func(inp):
                        x = inp
                        x = Conv2D(dim, kernel_size=3, padding='same')(x)
                        x = LeakyReLU(0.2)(x)
                        x = Conv2D(dim, kernel_size=3, padding='same')(x)
                        x = Add()([x, inp])
                        x = LeakyReLU(0.2)(x)
                        return x
                    return func

                def to_bgr (output_nc):
                    def func(x):
                        return Conv2D(output_nc, kernel_size=1, padding='valid',  activation='tanh')(x)#, activation='sigmoid'
                    return func

                def enc_flow(e_dims):
                    def func(x):                        
                        for i in range(5):
                            x = LeakyReLU(0.1)(Conv2D(e_dims * 2**i, kernel_size=5, strides=2, padding='same')(x))
                            x = ResidualBlock(e_dims * 2**i )(x)
                            
                        x = Conv2D( e_dims * 2**5 , kernel_size=5, strides=2, padding='same')(x)
                        return x
                    return func

                def dec_x_base(ae_dims, d_dims):
                    def func(x):
                        x = Dense(ae_dims)(Flatten()(x))
                        x = Dense(4 * 4 * ae_dims)(x)
                        x = Reshape((4, 4, ae_dims))(x)                        
                        x = SubpixelUpscaler()(LeakyReLU(0.1)(Conv2D(ae_dims*4, kernel_size=3, strides=1, padding='same')(x)))
                        x = SubpixelUpscaler()(LeakyReLU(0.1)(Conv2D(ae_dims*4, kernel_size=3, strides=1, padding='same')(x)))
                        return x
                    return func
                
                def dec_x_flow(d_dims, by_diff=True):
                    def func(x):
                        #x = SubpixelUpscaler()(LeakyReLU(0.1)(Conv2D(d_dims*4, kernel_size=3, strides=1, padding='same')(x)))
                        #x = ResidualBlock(d_dims)(x)
                        #x = ResidualBlock(d_dims)(x)
                        #return x
                        if by_diff:
                            x2_inp = UpSampling2D()(x)
                        
                        x_inp = x = SubpixelUpscaler()(LeakyReLU(0.1)(Conv2D(d_dims*4, kernel_size=3, strides=1, padding='same')(x)))                            
                        x = Conv2D(d_dims, kernel_size=3, padding='same')(x)
                        x = LeakyReLU(0.2)(x)
                        x = Conv2D(d_dims, kernel_size=3, padding='same')(x)
                        x = Add()([x, x_inp])
                        
                        if by_diff:
                            x = Add()([x, x2_inp])                  
                        
                        x = LeakyReLU(0.2)(x)
                            
                        return x
                    return func
                    
                self.encoder = modelify(enc_flow(e_dims)) ( Input(bgr_shape) )
                
                sh = K.int_shape( self.encoder.outputs[0] )[1:]
                self.decoder_src_base = modelify(dec_x_base(ae_dims, d_dims)) ( Input(sh) )
                self.decoder_dst_base = modelify(dec_x_base(ae_dims, d_dims)) ( Input(sh) )
                
                self.decoder_src_x_ar = []
                self.decoder_dst_x_ar = []
                self.decoder_src_x_bgr_ar = []
                self.decoder_dst_x_bgr_ar = []
                self.decoder_src_x_m_ar = []
                self.decoder_dst_x_m_ar = []

                for i in range(upscale_count):
                    if i == 0:
                        inp_sh = K.int_shape( self.decoder_src_base.outputs[0] )[1:]
                        self.decoder_src_x_ar.append ( modelify(dec_x_flow(d_dims))(Input(inp_sh)) )
                        self.decoder_dst_x_ar.append ( modelify(dec_x_flow(d_dims))(Input(inp_sh)) )
                    else:
                        inp_sh = K.int_shape( self.decoder_src_x_ar[-1].outputs[0] )[1:]
                        self.decoder_src_x_ar.append ( modelify(dec_x_flow(d_dims))(Input(inp_sh)) )
                        self.decoder_dst_x_ar.append ( modelify(dec_x_flow(d_dims))(Input(inp_sh)) )

                    inp_sh = K.int_shape( self.decoder_src_x_ar[-1].outputs[0] )[1:]
                    self.decoder_src_x_bgr_ar.append ( modelify(to_bgr(3))( Input(inp_sh)) )
                    self.decoder_dst_x_bgr_ar.append ( modelify(to_bgr(3))( Input(inp_sh)) )
                    if learn_mask:
                        self.decoder_src_x_m_ar.append ( modelify(to_bgr(1))( Input(inp_sh)) )
                        self.decoder_dst_x_m_ar.append ( modelify(to_bgr(1))( Input(inp_sh)) )

                self.warped_src_t, self.warped_dst_t = Input(bgr_shape), Input(bgr_shape)
                src_code, dst_code = self.encoder(self.warped_src_t), self.encoder(self.warped_dst_t)

                self.pred_src_src = []
                self.pred_src_srcm = []

                self.pred_dst_dst = []
                self.pred_dst_dstm = []

                self.pred_src_dst = []
                self.pred_src_dstm = []

                x_src = self.decoder_src_base(src_code)
                x_dst = self.decoder_dst_base(dst_code)
                x_src_dst = self.decoder_src_base(dst_code)
                
                for i in range(upscale_count):

                    x_src = self.decoder_src_x_ar[i]     (x_src)
                    self.pred_src_src += [self.decoder_src_x_bgr_ar[i] (x_src) ]
                    if learn_mask:
                        self.pred_src_srcm += [ self.decoder_src_x_m_ar[i](x_src) ]

                    x_dst = self.decoder_dst_x_ar[i]     (x_dst)
                    self.pred_dst_dst += [self.decoder_dst_x_bgr_ar[i] (x_dst)]
                    if learn_mask:
                        self.pred_dst_dstm += [ self.decoder_dst_x_m_ar[i](x_dst), ]

                    x_src_dst = self.decoder_src_x_ar[i]     (x_src_dst)
                    self.pred_src_dst += [self.decoder_src_x_bgr_ar[i] (x_src_dst) ]
                    if learn_mask:
                        self.pred_src_dstm += [ self.decoder_src_x_m_ar[i](x_src_dst), ]

            def get_model_filename_list(self):
                return [ [self.encoder, 'encoder.h5'] ] + \
                       [ [self.decoder_src_base, 'decoder_src_base.h5'] ] + \
                       [ [self.decoder_dst_base, 'decoder_dst_base.h5'] ] + \
                       [ [self.decoder_src_x_ar[i], 'decoder_src_x_%d.h5' % i ] for i in range(self.upscale_count)] + \
                       [ [self.decoder_src_x_bgr_ar[i], 'decoder_src_x_bgr_%d.h5' % i ] for i in range(self.upscale_count)] + \
                       [ [self.decoder_dst_x_ar[i], 'decoder_dst_x_%d.h5' % i ] for i in range(self.upscale_count)] + \
                       [ [self.decoder_dst_x_bgr_ar[i], 'decoder_dst_x_bgr_%d.h5' % i ] for i in range(self.upscale_count)]

        class SAELIAEModel(object):
            def __init__(self, resolution, ae_dims, e_dims, d_dims, learn_mask, upscale_count=3):
                super().__init__()
                self.resolution = resolution
                self.ae_dims = ae_dims
                self.e_dims = e_dims
                self.d_dims = d_dims
                self.learn_mask = learn_mask
                self.upscale_count = upscale_count

                bgr_shape = (resolution, resolution, 3)
                mask_shape = (resolution, resolution, 1)

                def ResidualBlock(dim, kernel_size=3):
                    def func(inp):
                        x = inp
                        x = Conv2D(dim, kernel_size=kernel_size, padding='same')(x)
                        x = LeakyReLU(0.2)(x)
                        x = Conv2D(dim, kernel_size=kernel_size, padding='same')(x)
                        x = Add()([x, inp])
                        x = LeakyReLU(0.2)(x)
                        return x
                    return func

                def downscale (dim):
                    def func(x):
                        return LeakyReLU(0.1)(Conv2D(dim, kernel_size=3, strides=2, padding='same')(x))
                    return func

                def to_bgr (output_nc):
                    def func(x):
                        return Conv2D(output_nc, kernel_size=1, padding='valid', activation='tanh')(x)
                    return func
                def to_mask (output_nc):
                    def func(x):
                        return Conv2D(output_nc, kernel_size=1, padding='valid', activation='sigmoid')(x)
                    return func

                def enc_flow(e_dims, ae_dims):
                    def func(x):
                        x = downscale(e_dims)(x)
                        x = downscale(e_dims*2)(x)
                        x = downscale(e_dims*4)(x)
                        x = downscale(e_dims*8)(x)
                        x = Flatten()(x)
                        return x
                    return func

                def inter_flow(ae_dims):
                    def func(x):
                        x = Dense(ae_dims)(x)
                        x = Dense(7 * 7 * ae_dims)(x)
                        x = Reshape((7, 7, ae_dims))(x)
                        x = SubpixelUpscaler()(LeakyReLU(0.1)(Conv2D(ae_dims*4, kernel_size=3, strides=1, padding='same')(x)))
                        
                        return x
                    return func
                    
                def dec_base_flow(ae_dims, d_dims):
                    def func(x):                        
                        x = SubpixelUpscaler()(LeakyReLU(0.1)(Conv2D(d_dims*4, kernel_size=3, strides=1, padding='same')(x)))                        
                        return x
                    return func

                def dec_x_flow(d_dims, by_diff=True):
                    def func(x):
                        if by_diff:
                            x2_inp = UpSampling2D()(x)
                        
                        x_inp = x = SubpixelUpscaler()(LeakyReLU(0.1)(Conv2D(d_dims*4, kernel_size=3, strides=1, padding='same')(x)))                            
                        x = Conv2D(d_dims, kernel_size=3, padding='same')(x)
                        x = LeakyReLU(0.2)(x)
                        x = Conv2D(d_dims, kernel_size=3, padding='same')(x)
                        x = Add()([x, x_inp])

                        if by_diff:
                            x = Add()([x, x2_inp])
                        else:
                            x = LeakyReLU(0.2)(x)
                            
                        return x
                    return func

                self.encoder = modelify(enc_flow(e_dims, ae_dims)) ( Input(bgr_shape) )

                self.inter_B  = modelify(inter_flow(ae_dims)) ( Input( K.int_shape( self.encoder.outputs[0] )[1:] ) )
                self.inter_AB = modelify(inter_flow(ae_dims)) ( Input( K.int_shape( self.encoder.outputs[0] )[1:] ) )
                
                sh = np.array(K.int_shape( self.inter_B.outputs[0] )[1:])*(1,1,2)

                self.dec_base = modelify(dec_base_flow(ae_dims,d_dims)) ( Input(sh) )
                
                self.decoder_x_ar = []
                self.decoder_x_bgr_ar = []
                self.decoder_x_m_ar = []

                for i in range(upscale_count):
                    if i == 0:
                        inp_sh = K.int_shape( self.dec_base.outputs[0] )[1:]
                        self.decoder_x_ar.append ( modelify(dec_x_flow( d_dims ))(Input(inp_sh)) )
                    else:
                        inp_sh = K.int_shape( self.decoder_x_ar[-1].outputs[0] )[1:]
                        self.decoder_x_ar.append ( modelify(dec_x_flow( d_dims ))(Input(inp_sh)) )

                    inp_sh = K.int_shape( self.decoder_x_ar[-1].outputs[0] )[1:]
                    self.decoder_x_bgr_ar.append ( modelify(to_bgr(3))( Input(inp_sh)) )
                    if learn_mask:
                        self.decoder_x_m_ar.append ( modelify(to_mask(1))( Input(inp_sh)) )

                self.warped_src_t, self.warped_dst_t = Input(bgr_shape), Input(bgr_shape)

                src_code, dst_code = self.encoder(self.warped_src_t), self.encoder(self.warped_dst_t)

                src_inter_AB_code = self.inter_AB(src_code)
                src_inter_code = Concatenate()([src_inter_AB_code,src_inter_AB_code])

                dst_inter_B_code  = self.inter_B (dst_code)
                dst_inter_AB_code = self.inter_AB (dst_code)
                dst_inter_code = Concatenate()([dst_inter_B_code,dst_inter_AB_code])

                src_dst_inter_code = Concatenate()([dst_inter_AB_code,dst_inter_AB_code])

                self.pred_src_src = []
                self.pred_src_srcm = []

                self.pred_dst_dst = []
                self.pred_dst_dstm = []

                self.pred_src_dst = []
                self.pred_src_dstm = []

                x_src = self.dec_base(src_inter_code)
                x_dst = self.dec_base(dst_inter_code)
                x_src_dst = self.dec_base(src_dst_inter_code)
                for i in range(upscale_count):

                    x_src = self.decoder_x_ar[i]     (x_src)
                    x_dst = self.decoder_x_ar[i]     (x_dst)
                    x_src_dst = self.decoder_x_ar[i]     (x_src_dst)

                    self.pred_src_src += [self.decoder_x_bgr_ar[i] (x_src)]
                    self.pred_dst_dst += [self.decoder_x_bgr_ar[i] (x_dst)]
                    self.pred_src_dst += [self.decoder_x_bgr_ar[i] (x_src_dst)]

                    if learn_mask:
                        self.pred_src_srcm += [ self.decoder_x_m_ar[i](x_src)]
                        self.pred_dst_dstm += [ self.decoder_x_m_ar[i](x_dst)]
                        self.pred_src_dstm += [ self.decoder_x_m_ar[i](x_src_dst)]

            def get_model_filename_list(self):
                result = [ [self.encoder, 'encoder.h5'] ] + \
                         [ [self.inter_B, 'inter_B.h5'] ] + \
                         [ [self.inter_AB, 'inter_AB.h5'] ] + \
                         [ [self.dec_base, 'dec_base.h5'] ] + \
                         [ [self.decoder_x_ar[i], 'decoder_x_%d.h5' % i ] for i in range(self.upscale_count)] + \
                         [ [self.decoder_x_bgr_ar[i], 'decoder_x_bgr_%d.h5' % i ] for i in range(self.upscale_count) ]

                if self.learn_mask:
                    result += [ [self.decoder_x_m_ar[i], 'decoder_x_m_%d.h5' % i ] for i in range(self.upscale_count) ]

                return result

        if 'df' in self.options['archi']:
            self.model = SAEDFModel(resolution=resolution, ae_dims=ae_dims, e_dims=e_dims, d_dims=d_dims, learn_mask=self.options['learn_mask'], upscale_count=self.upscale_count )
        elif 'liae' in self.options['archi']:
            self.model = SAELIAEModel(resolution=resolution, ae_dims=ae_dims, e_dims=e_dims, d_dims=d_dims, learn_mask=self.options['learn_mask'], upscale_count=self.upscale_count )


        if self.is_first_run():
            if self.options.get('ca_weights',False):
                conv_weights_list = []
                for model, _ in self.model.get_model_filename_list():
                    for layer in model.layers:
                        if type(layer) == keras.layers.Conv2D:
                            conv_weights_list += [layer.weights[0]] #Conv2D kernel_weights
                CAInitializerMP ( conv_weights_list )
        else:
            self.load_weights_safe( self.model.get_model_filename_list() )

        target_src = Input ( stage_bgr_shape )
        target_srcm = Input ( stage_mask_shape )
        target_dst = Input ( stage_bgr_shape )
        target_dstm = Input ( stage_mask_shape )

        target_srcm_sigm = gaussian_blur( max(1, K.int_shape( target_srcm )[1] // 32) )(target_srcm)
        target_srcm_anti_sigm = 1.0 - target_srcm_sigm

        target_dstm_sigm = gaussian_blur( max(1, K.int_shape( target_dstm )[1] // 32) )(target_dstm)
        target_dstm_anti_sigm = 1.0 - target_dstm_sigm

        if stage == 0:
            pred_src_src = self.model.pred_src_src[0]
            pred_dst_dst = self.model.pred_dst_dst[0]
            pred_src_dst = self.model.pred_src_dst[0]

            if self.options['learn_mask']:
                pred_src_srcm = self.model.pred_src_srcm[0]
                pred_dst_dstm = self.model.pred_dst_dstm[0]
                pred_src_dstm = self.model.pred_src_dstm[0]
        else:
            alpha = (self.iter - self.start_stage_iter) / ( self.target_stage_iter - self.start_stage_iter )
            alpha = np.clip(alpha, 0, 1)
            
            self.ws_layer = WeightedSum()
            self.ws_layer.set_alpha(alpha)

            if self.options['learn_mask']:
                self.wsm_layer = WeightedSum()
                self.wsm_layer.set_alpha(alpha)
                
            pred_src_src = self.ws_layer([ UpSampling2D()(self.model.pred_src_src[stage-1]), self.model.pred_src_src[stage] ] )
            pred_dst_dst = self.ws_layer([ UpSampling2D()(self.model.pred_dst_dst[stage-1]), self.model.pred_dst_dst[stage] ] )
            pred_src_dst = self.ws_layer([ UpSampling2D()(self.model.pred_src_dst[stage-1]), self.model.pred_src_dst[stage] ] )

            if self.options['learn_mask']:
                pred_src_srcm = self.wsm_layer([ UpSampling2D()(self.model.pred_src_srcm[stage-1]), self.model.pred_src_srcm[stage] ] )
                pred_dst_dstm = self.wsm_layer([ UpSampling2D()(self.model.pred_dst_dstm[stage-1]), self.model.pred_dst_dstm[stage] ] )
                pred_src_dstm = self.wsm_layer([ UpSampling2D()(self.model.pred_src_dstm[stage-1]), self.model.pred_src_dstm[stage] ] )


        if 'df' in self.options['archi']:
            trainable_weights = self.model.encoder.trainable_weights
            for i in range (stage+1):
                trainable_weights += self.model.decoder_src_x_ar[i].trainable_weights
                trainable_weights += self.model.decoder_dst_x_ar[i].trainable_weights

            trainable_weights += self.model.decoder_src_x_bgr_ar[stage].trainable_weights
            trainable_weights += self.model.decoder_dst_x_bgr_ar[stage].trainable_weights
        elif 'liae' in self.options['archi']:

            trainable_weights = self.model.encoder.trainable_weights + \
                                self.model.inter_B.trainable_weights + \
                                self.model.inter_AB.trainable_weights

            if self.options['learn_mask']:
                src_dst_mask_loss_train_weights = self.model.encoder.trainable_weights + \
                                                    self.model.inter_B.trainable_weights + \
                                                    self.model.inter_AB.trainable_weights

            for i in range (stage+1):
                trainable_weights += self.model.decoder_x_ar[i].trainable_weights
                if self.options['learn_mask']:
                    src_dst_mask_loss_train_weights += self.model.decoder_x_ar[i].trainable_weights

            trainable_weights += self.model.decoder_x_bgr_ar[stage].trainable_weights
            if self.options['learn_mask']:
                src_dst_mask_loss_train_weights += self.model.decoder_x_m_ar[stage].trainable_weights


        if self.is_training_mode:
            self.src_dst_opt      = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999, clipnorm=1.0 if self.options['clipgrad'] else 0.0, tf_cpu_mode=self.options['optimizer_mode']-1)
            self.src_dst_mask_opt = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999, clipnorm=1.0 if self.options['clipgrad'] else 0.0, tf_cpu_mode=self.options['optimizer_mode']-1)

            if not pixel_loss:
                src_loss_batch = 10*dssim(kernel_size=int(stage_resolution/11.6),max_value=2.0)( (target_src+1)*target_srcm_sigm, (pred_src_src+1)*target_srcm_sigm )
            else:
                src_loss_batch = K.mean ( 50*K.square( (target_src+1)*target_srcm_sigm - (pred_src_src+1)*target_srcm_sigm), axis=[1,2,3])

            src_loss = K.mean(src_loss_batch)

            face_style_power = self.options['face_style_power']  / 100.0

            if face_style_power != 0:
                src_loss += style_loss(gaussian_blur_radius=stage_resolution//16, loss_weight=face_style_power, wnd_size=0)( ((pred_src_dst+1)/2)*target_dstm_sigm, ((target_dst+1)/2)*target_dstm_sigm )

            bg_style_power = self.options['bg_style_power'] / 100.0
            if bg_style_power != 0:
                if not pixel_loss:
                    bg_loss = K.mean( (10*bg_style_power)*dssim(kernel_size=int(stage_resolution/11.6),max_value=2.0)( (pred_src_dst+1)*target_dstm_anti_sigm, (target_dst+1)*target_dstm_anti_sigm ))
                else:
                    bg_loss = K.mean( (50*bg_style_power)*K.square( (pred_src_dst+1)*target_dstm_anti_sigm - (target_dst+1)*target_dstm_anti_sigm ))
                src_loss += bg_loss

            if not pixel_loss:
                dst_loss_batch = 10*dssim(kernel_size=int(stage_resolution/11.6),max_value=2.0)( (target_dst+1)*target_dstm_sigm, (pred_dst_dst+1)*target_dstm_sigm )
            else:
                dst_loss_batch = K.mean ( 50*K.square( (target_dst+1)*target_dstm_sigm - (pred_dst_dst+1)*target_dstm_sigm ), axis=[1,2,3])

            dst_loss = K.mean(dst_loss_batch)

            feed = [self.model.warped_src_t, target_src, target_srcm, self.model.warped_dst_t, target_dst, target_dstm]


            self.src_dst_train = K.function (feed,[src_loss,dst_loss], self.src_dst_opt.get_updates(src_loss+dst_loss, trainable_weights ) )


            if self.options['learn_mask']:
                src_mask_loss = K.mean(K.square(target_srcm-pred_src_srcm))
                dst_mask_loss = K.mean(K.square(target_dstm-pred_dst_dstm))

                feed = [self.model.warped_src_t, target_srcm, self.model.warped_dst_t, target_dstm]

                self.src_dst_mask_train = K.function (feed, [src_mask_loss, dst_mask_loss], self.src_dst_mask_opt.get_updates(src_mask_loss+dst_mask_loss, src_dst_mask_loss_train_weights) )

            if self.options['learn_mask']:
                self.AE_view = K.function ([ self.model.warped_src_t, self.model.warped_dst_t ], [pred_src_src, pred_dst_dst, pred_dst_dstm, pred_src_dst, pred_src_dstm])
            else:
                self.AE_view = K.function ([ self.model.warped_src_t, self.model.warped_dst_t ], [pred_src_src, pred_dst_dst, pred_src_dst ] )


        else:
            if self.options['learn_mask']:
                self.AE_convert = K.function ([self.model.warped_dst_t],[ pred_src_dst, pred_dst_dstm, pred_src_dstm ])
            else:
                self.AE_convert = K.function ([self.model.warped_dst_t],[ pred_src_dst ])


        if self.is_training_mode:
            self.src_sample_losses = []
            self.dst_sample_losses = []

            t = SampleProcessor.Types
            face_type = t.FACE_TYPE_FULL if self.options['face_type'] == 'f' else t.FACE_TYPE_HALF

            t_mode_bgr = t.MODE_BGR if not self.pretrain else t.MODE_BGR_SHUFFLE

            training_data_src_path = self.training_data_src_path
            training_data_dst_path = self.training_data_dst_path
            sort_by_yaw = self.sort_by_yaw

            if self.pretrain and self.pretraining_data_path is not None:
                training_data_src_path = self.pretraining_data_path
                training_data_dst_path = self.pretraining_data_path
                sort_by_yaw = False

            self.set_training_data_generators ([
                    SampleGeneratorFace(training_data_src_path, sort_by_yaw_target_samples_path=training_data_dst_path if sort_by_yaw else None,
                                                                random_ct_samples_path=training_data_dst_path if apply_random_ct else None,
                                                                debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(random_flip=self.random_flip, scale_range=np.array([-0.05, 0.05])+self.src_scale_mod / 100.0 ),
                        output_sample_types = [ {'types' : (t.IMG_WARPED_TRANSFORMED, face_type, t_mode_bgr), 'resolution':resolution, 'apply_ct': apply_random_ct, 'normalize_tanh':True },
                                                {'types' : (t.IMG_TRANSFORMED, face_type, t_mode_bgr), 'resolution': stage_resolution, 'apply_ct': apply_random_ct, 'normalize_tanh':True },
                                                {'types' : (t.IMG_TRANSFORMED, face_type, t.MODE_M), 'resolution': stage_resolution },
                                                {'types' : (t.IMG_TRANSFORMED, face_type, t_mode_bgr), 'resolution': resolution, 'normalize_tanh':True},
                                              ]
                         ),

                    SampleGeneratorFace(training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(random_flip=self.random_flip, ),
                        output_sample_types = [ {'types' : (t.IMG_WARPED_TRANSFORMED, face_type, t_mode_bgr), 'resolution':resolution, 'normalize_tanh':True},
                                                {'types' : (t.IMG_TRANSFORMED, face_type, t_mode_bgr), 'resolution': stage_resolution, 'normalize_tanh':True},
                                                {'types' : (t.IMG_TRANSFORMED, face_type, t.MODE_M), 'resolution': stage_resolution},
                                                {'types' : (t.IMG_TRANSFORMED, face_type, t_mode_bgr), 'resolution': resolution, 'normalize_tanh':True},
                                              ])
                    ])

    #override
    def get_model_filename_list(self):
        return self.model.get_model_filename_list()

    #override
    def onSave(self):
        self.save_weights_safe( self.get_model_filename_list() )

    #override
    def onTrainOneIter(self, generators_samples, generators_list):
        warped_src, target_src, target_srcm, target_src_res = generators_samples[0]
        warped_dst, target_dst, target_dstm, target_dst_res = generators_samples[1]

        if self.stage != 0:
            alpha = (self.iter - self.start_stage_iter) / ( self.target_stage_iter - self.start_stage_iter )
            alpha = np.clip(alpha, 0, 1)
            self.ws_layer.set_alpha(alpha)

            if self.options['learn_mask']:
                self.wsm_layer.set_alpha(alpha)

        src_loss, dst_loss, = self.src_dst_train ( [ warped_src, target_src, target_srcm, warped_dst, target_dst, target_dstm ])

        if self.options['learn_mask']:
            src_mask_loss, dst_mask_loss, = self.src_dst_mask_train ([warped_src, target_srcm, warped_dst, target_dstm])

        return ( ('src_loss', src_loss), ('dst_loss', dst_loss) )


    #override
    def onGetPreview(self, sample):
        nsamples = 2
        test_S   = sample[0][3][0:nsamples] #first nsamples samples
        test_S_m = sample[0][2][0:nsamples] #first nsamples samples
        test_D   = sample[1][3][0:nsamples]
        test_D_m = sample[1][2][0:nsamples]

        if self.options['learn_mask']:
            S, SM, D, DM, SS, DD, DDM, SD, SDM = [ np.clip( np.stack([ cv2.resize(  (y/2)+0.5, (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST ) for y in x ]) , 0.0, 1.0) for x in ([test_S,test_S_m,test_D, test_D_m] + self.AE_view ([test_S, test_D]) ) ]

            SM, DM, DDM, SDM, = [ np.repeat ( x[...,None], (3,), -1) for x in [SM, DM, DDM, SDM] ]
        else:
            S, D, SS, DD, SD, = [ np.clip( np.stack([ cv2.resize(  (y/2)+0.5, (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST ) for y in x ]) , 0.0, 1.0) for x in ([test_S,test_D] + self.AE_view ([test_S, test_D]) ) ]



        #S = np.stack([ cv2.resize(x, (self.stage_resolution, self.stage_resolution) ) for x in S ])
        #D = np.stack([ cv2.resize(x, (self.stage_resolution, self.stage_resolution) ) for x in D ])
        #import code
        #code.interact(local=dict(globals(), **locals()))

        result = []
        st = []
        for i in range(nsamples):
            ar = S[i], SS[i], D[i], DD[i], SD[i]
            st.append ( np.concatenate ( ar, axis=1) )

        result += [ ('PGSAE', np.concatenate (st, axis=0 )), ]

        if self.options['learn_mask']:
            st_m = []
            for i in range(nsamples):
                ar = S[i]*SM[i], SS[i], D[i]*DM[i], DD[i]*DDM[i], SD[i]*(DDM[i]*SDM[i])
                st_m.append ( np.concatenate ( ar, axis=1) )

            result += [ ('PGSAE masked', np.concatenate (st_m, axis=0 )), ]

        return result

    def predictor_func (self, face=None, dummy_predict=False):
        if dummy_predict:
            self.AE_convert ([ np.zeros ( (1, self.resolution, self.resolution, 3), dtype=np.float32 ) ])
        else:
            face = face * 2 - 1
            if self.options['learn_mask']:
                bgr, mask_dst_dstm, mask_src_dstm = self.AE_convert ([face[np.newaxis,...]])
                mask = mask_dst_dstm[0] * mask_src_dstm[0]
                return bgr[0], mask[...,0]
            else:
                bgr, = self.AE_convert ([face[np.newaxis,...]])
                bgr = cv2.resize ( (bgr[0] / 2) + 0.5, (self.resolution, self.resolution), interpolation=cv2.INTER_LANCZOS4 )

                return bgr

    #override
    def get_ConverterConfig(self):
        face_type = FaceType.FULL if self.options['face_type'] == 'f' else FaceType.HALF

        import converters
        return self.predictor_func, (self.resolution, self.resolution, 3), converters.ConverterConfigMasked(
                                     face_type=face_type,
                                     default_mode = 1 if self.options['apply_random_ct'] or self.options['face_style_power'] or self.options['bg_style_power'] else 4,                                     
                                     clip_hborder_mask_per=0.0625 if (self.options['face_type'] == 'f') else 0,
                                    )


Model = SAEModel

"""
class SAELIAEModel(object):
            def __init__(self, resolution, ae_dims, e_dims, d_dims, learn_mask, upscale_count=3):
                super().__init__()
                self.resolution = resolution
                self.ae_dims = ae_dims
                self.e_dims = e_dims
                self.d_dims = d_dims
                self.learn_mask = learn_mask
                self.upscale_count = upscale_count

                bgr_shape = (resolution, resolution, 3)
                mask_shape = (resolution, resolution, 1)

                def ResidualBlock(dim, kernel_size=3):
                    def func(inp):
                        x = inp
                        x = Conv2D(dim, kernel_size=kernel_size, padding='same')(x)
                        x = LeakyReLU(0.2)(x)
                        x = Conv2D(dim, kernel_size=kernel_size, padding='same')(x)
                        x = Add()([x, inp])
                        x = LeakyReLU(0.2)(x)
                        return x
                    return func

                def downscale (dim):
                    def func(x):
                        return LeakyReLU(0.1)(Conv2D(dim, kernel_size=3, strides=2, padding='same')(x))
                    return func

                def to_bgr (output_nc):
                    def func(x):
                        return Conv2D(output_nc, kernel_size=1, padding='valid', activation='tanh')(x)
                    return func
                def to_mask (output_nc):
                    def func(x):
                        return Conv2D(output_nc, kernel_size=1, padding='valid', activation='sigmoid')(x)
                    return func

                def enc_flow(e_dims, ae_dims):
                    def func(x):
                        
                        for i in range(3):
                            x = downscale(e_dims * 2**i )(x)
                            x = ResidualBlock(e_dims * 2**i )(x)
                            x = ResidualBlock(e_dims * 2**i )(x)
                        
                        x = Conv2D( e_dims * 2**3 , kernel_size=3, strides=2, padding='same')(x)
                        return x
                    return func

        
                def inter_flow(ae_dims):
                    def func(x):
                        x = Dense(ae_dims)(x)
                        x = Dense(7 * 7 * ae_dims)(x)
                        x = Reshape((7, 7, ae_dims))(x)
                        x = SubpixelUpscaler()(LeakyReLU(0.1)(Conv2D(ae_dims*4, kernel_size=3, strides=1, padding='same')(x)))
                        
                        return x
                    return func
                    
                def dec_base_flow(ae_dims, d_dims):
                    def func(x):                        
                        x = SubpixelUpscaler()(LeakyReLU(0.1)(Conv2D(d_dims*4, kernel_size=3, strides=1, padding='same')(x)))                        
                        return x
                    return func

                def dec_x_flow(d_dims, by_diff=True):
                    def func(x):
                        if by_diff:
                            x2_inp = UpSampling2D()(x)
                        
                        x_inp = x = SubpixelUpscaler()(LeakyReLU(0.1)(Conv2D(d_dims*4, kernel_size=3, strides=1, padding='same')(x)))                            
                        x = Conv2D(d_dims, kernel_size=3, padding='same')(x)
                        x = LeakyReLU(0.2)(x)
                        x = Conv2D(d_dims, kernel_size=3, padding='same')(x)
                        x = Add()([x, x_inp])

                        if by_diff:
                            x = Add()([x, x2_inp])
                        else:
                            x = LeakyReLU(0.2)(x)
                            
                        return x
                    return func
         

                self.encoder = modelify(enc_flow(e_dims, ae_dims)) ( Input(bgr_shape) )

                #self.inter_B  = modelify(inter_flow(ae_dims)) ( Input( K.int_shape( self.encoder.outputs[0] )[1:] ) )
                #self.inter_AB = modelify(inter_flow(ae_dims)) ( Input( K.int_shape( self.encoder.outputs[0] )[1:] ) )
                
                sh = np.array(K.int_shape( self.inter_B.outputs[0] )[1:])*(1,1,2)

                self.dec_base = modelify(dec_base_flow(ae_dims,d_dims)) ( Input(sh) )
                
                self.decoder_x_ar = []
                self.decoder_x_bgr_ar = []
                self.decoder_x_m_ar = []

                for i in range(upscale_count):
                    if i == 0:
                        inp_sh = K.int_shape( self.dec_base.outputs[0] )[1:]
                        self.decoder_x_ar.append ( modelify(dec_x_flow( d_dims ))(Input(inp_sh)) )
                    else:
                        inp_sh = K.int_shape( self.decoder_x_ar[-1].outputs[0] )[1:]
                        self.decoder_x_ar.append ( modelify(dec_x_flow( d_dims ))(Input(inp_sh)) )

                    inp_sh = K.int_shape( self.decoder_x_ar[-1].outputs[0] )[1:]
                    self.decoder_x_bgr_ar.append ( modelify(to_bgr(3))( Input(inp_sh)) )
                    if learn_mask:
                        self.decoder_x_m_ar.append ( modelify(to_mask(1))( Input(inp_sh)) )

                self.warped_src_t, self.warped_dst_t = Input(bgr_shape), Input(bgr_shape)

                src_code, dst_code = self.encoder(self.warped_src_t), self.encoder(self.warped_dst_t)

                
                src_inter_AB_code = self.inter_AB(src_code)
                src_inter_code = Concatenate()([src_inter_AB_code,src_inter_AB_code])

                dst_inter_B_code  = self.inter_B (dst_code)
                dst_inter_AB_code = self.inter_AB (dst_code)
                dst_inter_code = Concatenate()([dst_inter_B_code,dst_inter_AB_code])

                src_dst_inter_code = Concatenate()([dst_inter_AB_code,dst_inter_AB_code])
                
                x_src = self.dec_base(src_inter_code)
                x_dst = self.dec_base(dst_inter_code)
                x_src_dst = self.dec_base(src_dst_inter_code)
                

                self.pred_src_src = []
                self.pred_src_srcm = []

                self.pred_dst_dst = []
                self.pred_dst_dstm = []

                self.pred_src_dst = []
                self.pred_src_dstm = []

                
                for i in range(upscale_count):

                    x_src = self.decoder_x_ar[i]     (x_src)
                    x_dst = self.decoder_x_ar[i]     (x_dst)
                    x_src_dst = self.decoder_x_ar[i]     (x_src_dst)

                    self.pred_src_src += [self.decoder_x_bgr_ar[i] (x_src)]
                    self.pred_dst_dst += [self.decoder_x_bgr_ar[i] (x_dst)]
                    self.pred_src_dst += [self.decoder_x_bgr_ar[i] (x_src_dst)]

                    if learn_mask:
                        self.pred_src_srcm += [ self.decoder_x_m_ar[i](x_src)]
                        self.pred_dst_dstm += [ self.decoder_x_m_ar[i](x_dst)]
                        self.pred_src_dstm += [ self.decoder_x_m_ar[i](x_src_dst)]

            def get_model_filename_list(self):
                
                result = [ [self.encoder, 'encoder.h5'] ] + \                        
                         [ [self.dec_base, 'dec_base.h5'] ] + \
                         [ [self.decoder_x_ar[i], 'decoder_x_%d.h5' % i ] for i in range(self.upscale_count)] + \
                         [ [self.decoder_x_bgr_ar[i], 'decoder_x_bgr_%d.h5' % i ] for i in range(self.upscale_count) ]

                if self.learn_mask:
                    result += [ [self.decoder_x_m_ar[i], 'decoder_x_m_%d.h5' % i ] for i in range(self.upscale_count) ]

                return result
"""