

class get_config():
  def __init__(self):
    # Transformer Parameters
    self.d_model=512  # Embedding Size
    self.channel_size = 192
    self.time_size = 201
    self.tokens = 10
    self.n_layers = 2 # number of Encoder of Decoder Layer
    self.n_classes = 26
    
    
    # Train Parameters
    self.epochs = 700
    self.batch_size_train = 8
    self.batch_size_test = 8
    self.epochs =500
    
    # Save Weights and load weight
    self.save_weights_path = './IC_512/'
    self.finish_weights_path = '/home/yzm/PycharmProjects/EEG-TCT/TCT/IC_checkpoints_512/EEGImaginedCharacter_Transformer_435_99.4482_95.7480_weights.pth'
    self.data_path = '../data/Character_imagine/character_imagine_1-process_10-26.mat'