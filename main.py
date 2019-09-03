import training
import unet_2d
import voc_reader

def main(argv=None):
    print("begin:")
    model=unet_2d.unet_model_2d_attention([256,256,3],21,batch_normalization=True)
    voc=voc_reader.voc_reader(256,256,8,8)
    batch_size=8
    steps_per_epoch=1464//batch_size
    validation_steps=1449//batch_size
    training.train_model(model,"model_File",training.train_generator_data(voc),training.val_generator_data(voc),steps_per_epoch,validation_steps)


if __name__=='__main__':
    main()