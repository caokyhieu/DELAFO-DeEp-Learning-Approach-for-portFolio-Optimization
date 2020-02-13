from keras import backend as K
from keras.layers import Layer
from keras import regularizers

class AdditiveAttentionLayer(Layer):

	def __init__(self, latent_dim=32,kernel_regularizer = None,
					**kwargs):
		self.latent_dim = latent_dim
		self.kernel_regularizer = regularizers.get(kernel_regularizer)
		super(AdditiveAttentionLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		in_seq_shape = input_shape[0]
		out_shape = input_shape[1]
		latent_dim = 64
		# Create a trainable weight variable for this layer.
		self.Wa = self.add_weight(name='Wa',
									  shape=(in_seq_shape[-1] , self.latent_dim),
									  initializer='uniform',
									  regularizer=self.kernel_regularizer,
									  trainable=True)
		self.Ua = self.add_weight(name='Ua',
									  shape=(out_shape[1], self.latent_dim),
									  initializer='uniform',
									  regularizer=self.kernel_regularizer,
									  trainable=True)
		self.Va = self.add_weight(name='Va',
									  shape=(self.latent_dim, 1),
									  initializer='uniform',
									  regularizer=self.kernel_regularizer,
									  trainable=True)
		super(AdditiveAttentionLayer, self).build(input_shape)  # Be sure to call this at the end

	def call(self, inputs):
		in_seq = inputs[0]
		out_vec = inputs[1]
		out_vec_shape = K.shape(out_vec)

		def cal_prob(out):
		  ## reshape input sequence from(batchsize,timesteps,features) to (batchsize*timesteps,features)
		  in_seq_shape = K.shape(in_seq)
		  in_seq_reshape = K.reshape(in_seq,(in_seq_shape[0]*in_seq_shape[1],-1))

		  ## Compute
		  W_as = K.dot(in_seq_reshape,self.Wa)
		  ##
		  out = K.dot(out,self.Ua)

		  out = K.reshape(K.repeat(out,n=in_seq_shape[1]),(in_seq_shape[0]*in_seq_shape[1],-1))


		  energy = K.reshape(K.dot(K.tanh(W_as+out),self.Va),(in_seq_shape[0],-1))

		  ## prob have shape(batchsize,timesteps)
		  prob = K.softmax(energy)
		  print('Shape of prob:')
		  print(K.int_shape(prob))

		  return prob

		def cal_contxt_vec(prob,in_seq):
		  contxt_vec =  K.sum(in_seq * K.expand_dims(prob,axis=-1),axis=1)

		  return contxt_vec

		prob = cal_prob(out_vec)
		contxt_vec = cal_contxt_vec(prob,in_seq)
		print('Shape of context vector:')
		print(K.int_shape(contxt_vec))

		return contxt_vec

	def compute_output_shape(self, input_shape):
		return (input_shape[0][0], input_shape[0][2])


class SelfAttentionLayer(Layer):

    def __init__(self,latent_dim=32,
                    kernel_regularizer = None,
                    **kwargs):
        self.latent_dim = latent_dim
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        # self.bias_regularizer = regularizers.get(bias_regularizer)
        super(SelfAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        timesteps = input_shape[1]
        h_dim = input_shape[2]
        # Create a trainable weight variable for this layer.
        self.WQ = self.add_weight(name='WQ',
                                      shape=(h_dim , self.latent_dim),
                                      initializer='uniform',
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)
        self.WK = self.add_weight(name='Ua',
                                      shape=(h_dim, self.latent_dim),
                                      initializer='uniform',
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)


#         self.Va = self.add_weight(name='Va',
#                                       shape=(latent_dim, 1),
#                                       initializer='uniform',
#                                       trainable=True)
        super(SelfAttentionLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        in_seq = inputs

        def cal_prob(inputs):
          ## reshape input sequence from(batchsize,timesteps,features) to (batchsize*timesteps,features)
          in_seq_shape = K.shape(in_seq)
#           in_seq_reshape = K.reshape(in_seq,(in_seq_shape[0]*in_seq_shape[1],-1))

          ## Compute
          query = K.dot(in_seq,self.WQ)
          ##
          key = K.dot(in_seq,self.WK)
          print(K.int_shape(key))

          energy = K.batch_dot(query,K.permute_dimensions(key,(0,2,1)))/self.latent_dim

        #   #### Apply masking prob here
        #   masking_prob = np.ones((K.int_shape(key)[1],K.int_shape(key)[1]))
        #   masking_prob = np.tril(masking_prob, k=0)

        #   ## apply masking to energy

        #   energy = energy * masking_prob

          ## prob have shape(batchsize,timesteps)
          prob = K.softmax(energy,axis=-1)
          print('Shape of prob:')
          print(K.int_shape(prob))

          return prob

        prob = cal_prob(in_seq)
#         out = K.batch_dot(prob,in_seq)

        return prob

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1],input_shape[1])
