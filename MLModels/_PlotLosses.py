from multiprocessing import current_process
if current_process().name == "MainProcess":
    import tensorflow as tf
    
import matplotlib.pyplot as plt

class PlotLosses(tf.keras.callbacks.Callback):
    
    def __init__(self, num = 0):
        self.num = num
    
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        plt.ion()
        plt.clf()
        plt.title(f'Step {self.num}-epoch:{epoch}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        pointer = -100 if self.i > 100 else 0
        
        plt.plot(self.x[pointer:], self.losses[pointer:], label="loss")
        plt.plot(self.x[pointer:], self.val_losses[pointer:], label = 'cv_loss')
        
        plt.legend()
        plt.grid(True, which = 'both')
        plt.draw()
        plt.pause(0.000001)
    
    def closePlot(self):
        plt.close()
