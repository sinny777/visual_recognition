
#!/usr/bin/env python

import tensorflow as tf
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)'
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2, VGG16, ResNet50, InceptionV3
from tensorflow.keras.models import Sequential

from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
# from keras.preprocessing import image
# from sklearn.utils import class_weight
from tensorflow.keras import optimizers

import math
from IPython.display import clear_output
from PIL import Image
# %matplotlib inline
import os
import datetime
import shutil

from handlers.data_handler import DataHandler, DataSet

class ModelHandler(object):
    def __init__(self, CONFIG):
        self.CONFIG = CONFIG
        self.NUM_CLASSES = 3
        self.data_handler = self.get_data_handler()

    def get_data_handler(self):
        return DataHandler(self.CONFIG)

    def clean(self):
        dirpath = self.CONFIG['MODEL_CONFIG']['log_dir']
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)
    
    def plot_history(self, H, NUM_EPOCHS ):
        plt.style.use("ggplot")
        fig = plt.figure()
        fig.set_size_inches(15, 5)
        
        fig.add_subplot(1, 3, 1)
        plt.plot(np.arange(0, NUM_EPOCHS), H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, NUM_EPOCHS), H.history["val_loss"], label="val_loss")
        plt.title("Training Loss and Validation Loss on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.legend(loc="lower left")

        
        fig.add_subplot(1, 3, 2)
        plt.plot(np.arange(0, NUM_EPOCHS), H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, NUM_EPOCHS), H.history["accuracy"], label="train_acc")
        plt.title("Training Loss and Accuracy on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        
        fig.add_subplot(1, 3, 3)
        plt.plot(np.arange(0, NUM_EPOCHS), H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, NUM_EPOCHS), H.history["val_accuracy"], label="val_acc")
        plt.title("Validation Loss and Accuracy on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")

        plt.show()
        #plt.savefig("plot.png")
    
    def get_callbacks(self):
        # Include the epoch in the file name (uses `str.format`)
        checkpoint_path = self.CONFIG['CHECKPOINTS_PATH']
        checkpoint_dir = os.path.dirname(checkpoint_path+'/cp-{epoch:04d}.ckpt')

        log_dir = self.CONFIG['MODEL_CONFIG']['log_dir'] + "/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        monitor = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            min_delta=1e-3, 
            patience=self.CONFIG["MODEL_CONFIG"]["patience"], 
            verbose=1, 
            restore_best_weights=True,
            mode='auto')
        # Create a callback that saves the model's weights every 5 epochs
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir, 
            monitor='val_acc',
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode='auto',
            save_freq=5)
        return [monitor, tensorboard_callback]

    def create_model(self, base_model_name):
        # self.data_dir = self.CONFIG['DATA_DIR']

        total_labels = len(self.data_handler.dataset.labels)
        print('total_labels: >> ', total_labels)

        my_new_model = Sequential()
        if  base_model_name == 'ResNet50':
            # resnet_weights_path = os.path.join(self.data_dir, 'pretrained_models/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
            # resnet = ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path)
            resnet = ResNet50(include_top=False, pooling='avg', weights='imagenet')            
            # resnet.summary()
            my_new_model.add(resnet)
            # Say no to train first layer (ResNet) model. It is already trained
            my_new_model.layers[0].trainable = False
        elif base_model_name == 'VGG16':
            # vgg_weights_path = os.path.join(self.data_dir, 'pretrained_models/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
            # vgg= VGG16(include_top=False, weights=vgg_weights_path ) 
            base_model= VGG16(include_top=False, weights='imagenet') 
            # vgg.summary()
            my_new_model.add(base_model)
            my_new_model.add(layers.GlobalAveragePooling2D())
            my_new_model.layers[0].trainable = False
            my_new_model.layers[1].trainable = False
        elif base_model_name == 'MobileNetV2':
            base_model= MobileNetV2(input_shape=(self.CONFIG['MODEL_CONFIG']['IMG_HEIGHT'], self.CONFIG['MODEL_CONFIG']['IMG_WIDTH'], 3), include_top=False, weights='imagenet')
            base_model.trainable = True
            fine_tune_at = 100
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable =  False
            # mobile_model.summary()
            my_new_model.add(base_model)
            my_new_model.add(layers.GlobalAveragePooling2D())
            my_new_model.add(layers.Dropout(rate=0.5))    
        elif base_model_name == 'InceptionV3':
            input_tensor = layers.Input(shape=(self.CONFIG['MODEL_CONFIG']['IMG_HEIGHT'], self.CONFIG['MODEL_CONFIG']['IMG_WIDTH'], 3))
            base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)
            my_new_model.add(base_model)
            my_new_model.add(layers.GlobalAveragePooling2D())
            my_new_model.add(layers.Dense(2048, activation='relu'))
            my_new_model.add(layers.Dropout(0.25))
            my_new_model.add(layers.Dense(1024, activation='relu'))
            my_new_model.add(layers.Dropout(0.2))  

            for layer in base_model.layers:
                layer.trainable = False

        else:

            data_augmentation = tf.keras.Sequential(
                [
                    layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                                input_shape=(self.CONFIG['MODEL_CONFIG']['IMG_HEIGHT'], self.CONFIG['MODEL_CONFIG']['IMG_WIDTH'], 3)),
                    layers.experimental.preprocessing.RandomRotation(0.1),
                    layers.experimental.preprocessing.RandomZoom(0.1),
                ]
            )

            my_new_model.add(data_augmentation)
            my_new_model.add(layers.experimental.preprocessing.Rescaling(1./255))
            my_new_model.add(layers.Conv2D(16, 3, padding='same', activation='relu'))
            my_new_model.add(layers.MaxPooling2D())
            my_new_model.add(layers.Conv2D(32, 3, padding='same', activation='relu'))
            my_new_model.add(layers.MaxPooling2D())
            my_new_model.add(layers.Conv2D(64, 3, padding='same', activation='relu'))
            my_new_model.add(layers.MaxPooling2D())
            my_new_model.add(layers.Dropout(0.2))
            my_new_model.add(layers.Flatten())
            my_new_model.add(layers.Dense(128, activation='relu'))
            
        my_new_model.add(layers.Dense(total_labels, activation=self.CONFIG['MODEL_CONFIG']['activation']))
   
        # opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
        # opt = tf.keras.optimizers.RMSprop(lr=2e-5)
        opt = tf.keras.optimizers.RMSprop()
        my_new_model.compile(optimizer=opt, loss=self.CONFIG['MODEL_CONFIG']['loss'], metrics=[self.CONFIG['MODEL_CONFIG']['metrics']])

        my_new_model.summary()

        return my_new_model
    
    def train_model(self, model, _epochs ):
        train_generator = self.data_handler.dataset.train_generator
        validation_generator = self.data_handler.dataset.validation_generator

        print('steps_per_epoch: >> ', self.data_handler.dataset._total_train//self.CONFIG['MODEL_CONFIG']['batch_size'])

        H = model.fit(
                train_generator,
                steps_per_epoch=self.data_handler.dataset._total_train//self.CONFIG['MODEL_CONFIG']['batch_size'],
                epochs=_epochs,
                validation_data=validation_generator,
                validation_steps=self.data_handler.dataset._total_val//self.CONFIG['MODEL_CONFIG']['batch_size']
                # callbacks = self.get_callbacks() #,
                #class_weight=dict_weights
                )
        
        return model, H
    
    def retrain_model(self, model, _epochs):
        #To train the top 2 inception blocks, freeze the first 249 layers and unfreeze the rest.
        for layer in model.layers[:249]:
            layer.trainable = False
        for layer in model.layers[249:]:
            layer.trainable = True
        #Recompile the model for these modifications to take effect
        opt = tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9)
        model.compile(optimizer=opt, loss=self.CONFIG['MODEL_CONFIG']['loss'], metrics=[self.CONFIG['MODEL_CONFIG']['metrics']])
        return self.train_model(model, _epochs)

    def save_model(self, model):
        model.save(self.CONFIG["MODEL_PATH"])
        # self.save_tflite_model()
        print("<<<<<<<< ML MODEL SAVED LOCALLY AT: ", self.CONFIG["MODEL_PATH"])
    
    def load_model(self):
        return load_model(self.CONFIG["MODEL_PATH"])
    
    def save_tflite_model(self):
        # train_generator, validation_generator = self.prepare_datasets()
        converter = tf.lite.TFLiteConverter.from_saved_model(self.CONFIG["MODEL_PATH"]) # path to the SavedModel directory
        tflite_model = converter.convert()
        # Show model size in KBs.
        float_model_size = len(tflite_model) / 1024
        print('Float model size = %dKBs.' % float_model_size)
        # Save the model.
        with open(self.CONFIG["MODEL_PATH"]+'/model.tflite', 'wb') as f:
            f.write(tflite_model)
            f.close()
        
        # Convert the model to the TensorFlow Lite format with quantization
        # def representative_dataset():
        #     for i in range(10):
        #         # xy = train_generator.next()
        #         image_batch, label_batch = next(train_generator)
        #         yield([image_batch[i].reshape(1, 1)])
        # Set the optimization flag.
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # Enforce integer only quantization
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # converter.inference_input_type = tf.int8
        # converter.inference_output_type = tf.int8
        # Provide a representative dataset to ensure we quantize correctly.
        # converter.representative_dataset = representative_dataset
        # converter.target_spec.supported_types = [tf.float32]
        model_tflite = converter.convert()
        with open(self.CONFIG["MODEL_PATH"]+'/model_lighter.tflite', 'wb') as f:
            f.write(model_tflite)
            f.close()


    def draw_prediction(self, frame, class_string ):
        x_start = frame.shape[1] -600
    #     cv2.putText(frame, class_string, (x_start, 75), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 0, 0), 2, cv2.LINE_AA)
        return frame
    
    def prepare_image_for_prediction(self, img):   
        # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
        # The below function inserts an additional dimension at the axis position provided
        img = np.expand_dims(img, axis=0)
        # perform pre-processing that was done when resnet model was trained.
        return preprocess_input(img)
    
    def get_display_string(self, pred_class, label_dict):
        txt = ""
        for c, confidence in pred_class:
            txt += label_dict[c]
            # if c :
            txt += '['+ str(confidence) +']'
        #print("count="+str(len(pred_class)) + " txt:" + txt)
        return txt
    
    def get_label_dict(self, train_generator ):
    # Get label to class_id mapping
        labels = (train_generator.class_indices)
        label_dict = dict((v,k) for k,v in labels.items())
        return  label_dict 
    
    def get_labels(self, generator ):
        generator.reset()
        labels = []
        for i in range(len(generator)):
            labels.extend(np.array(generator[i][1]) )
        return np.argmax(labels, axis =1)
    
    def get_pred_labels(self, test_generator):
        test_generator.reset()
        pred_vec=model.predict_generator(test_generator,
                                        steps=test_generator.n, #test_generator.batch_size
                                        verbose=1)
        return np.argmax( pred_vec, axis = 1), np.max(pred_vec, axis = 1)
    
    def predict_image(self, images):
        print("\n\nPrediction of Images using main model: ")

        label_dict = {0: 'default', 1: 'fire', 2: 'smoke'}
        model = self.load_model()
        predictions = []
        for image in images:
            # urllib.urlretrieve(image['path'], 'save.jpg') # or other way to upload image
            # img = load_img(image['path'], target_size=(self.CONFIG['MODEL_CONFIG']['IMG_HEIGHT'], self.CONFIG['MODEL_CONFIG']['IMG_WIDTH'])) # this is a PIL image 
            img = Image.open(image['path']).convert('RGB').resize((self.CONFIG['MODEL_CONFIG']['IMG_HEIGHT'], self.CONFIG['MODEL_CONFIG']['IMG_WIDTH']), Image.ANTIALIAS)
            # img = image.load_img(image['path'], target_size=(self.CONFIG['MODEL_CONFIG']['IMG_HEIGHT'], self.CONFIG['MODEL_CONFIG']['IMG_WIDTH']))
            frame_for_pred = img_to_array(img)
            frame_for_pred = np.expand_dims(frame_for_pred, axis=0) / 255
            # frame_for_pred = self.prepare_image_for_prediction( img )
            pred_vec = model.predict(frame_for_pred)
            # print(pred_vec)
            pred_class = []
            confidence = np.round(pred_vec.max(),2) 
            # confidence = pred_vec.max()
            # print('confidence: >> ', confidence)

            if confidence > 0.4:
                # pc = pred_vec.argmax()
                pc = np.argmax(pred_vec[0])
                pred_class.append( (pc, confidence) )
            else:
                pred_class.append( (0, 0) )
            if pred_class:
                # print('pred_class: >> ', pred_class)
                # print('confidence: >> ', confidence)
                result = self.get_display_string(pred_class, label_dict) 
                # print('Result: >> ', result)
                # print('\n')
            predictions.append({'name': image['name'], 'prediction': pred_class, 'result': result})
        return predictions

    def set_input_tensor(self, interpreter, image):
        tensor_index = interpreter.get_input_details()[0]['index']
        input_tensor = interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image
    
    def predict_tflite(self, images):
        print("\n\nPrediction of Images using light model: ")
        label_dict = {0: 'default', 1: 'fire', 2: 'smoke'}

        interpreter = tf.lite.Interpreter(model_path=self.CONFIG["MODEL_PATH"]+'/model_lighter.tflite')
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        predictions = []
        i = 0
        for image in images:
            # i = i + 1
            # if i == 20:
            #     break
           
            test_image = np.array(Image.open(image['path']).resize((self.CONFIG['MODEL_CONFIG']['IMG_HEIGHT'], self.CONFIG['MODEL_CONFIG']['IMG_WIDTH']))).astype(np.float32) / self.CONFIG['MODEL_CONFIG']['IMG_WIDTH'] - 1
            test_image = test_image.reshape(1,self.CONFIG['MODEL_CONFIG']['IMG_HEIGHT'],self.CONFIG['MODEL_CONFIG']['IMG_WIDTH'],3)

            # Adjust graph input to handle batch tensor
            # interpreter.resize_tensor_input(input_details[0]['index'], test_image.shape) #(batch_size, 512, 512, 3)

            # Adjust output #1 in graph to handle batch tensor
            # interpreter.resize_tensor_input(output_details[0]['index'], test_image.shape) #(batch_size, 512, 512, 3)
           
            # If required, quantize the input layer (from float to integer)
            input_scale, input_zero_point = input_details[0]["quantization"]
            if (input_scale, input_zero_point) != (0.0, 0):
                test_image = test_image / input_scale + input_zero_point
                test_image = test_image.astype(input_details[0]["dtype"])

            interpreter.set_tensor(input_details[0]["index"], test_image)
            interpreter.invoke()
            # print(output_details[0]) # Shape and shape signature are always (1, 512, 512, 3)        
            pred_vec = interpreter.get_tensor(output_details[0]["index"])
            # print(pred_vec)
            # If required, dequantized the output layer (from integer to float)
            output_scale, output_zero_point = output_details[0]["quantization"]
            if (output_scale, output_zero_point) != (0.0, 0):
                pred_vec = output_data.astype(np.float32)
                pred_vec = (pred_vec - output_zero_point) * output_scale
            
            # output = np.array(output_data)
            # output = np.squeeze(output_data[0])
            # print(output)

            pred_class =[]
            confidence = np.round(pred_vec.max(),2) 
            # confidence = pred_vec.max()
            # print('confidence: >> ', confidence)

            if confidence > 0.4:
                # pc = pred_vec.argmax()
                pc = np.argmax(pred_vec[0])
                pred_class.append( (pc, confidence) )
            else:
                pred_class.append( (0, 0) )
            if pred_class:
                # print('pred_class: >> ', pred_class)
                # print('confidence: >> ', confidence)
                result = self.get_display_string(pred_class, label_dict) 
                # print('Result: >> ', result)
                # print('\n')
            predictions.append({'name': image['name'], 'prediction': pred_class, 'result': result})
        return predictions

    def predict_tflite1(self, images):
        print("\n\nPrediction of Images using light model: ")
        label_dict = {0: 'fire', 1: 'default'}

        interpreter = None
        predictions = []
        i = 0
        for image in images:
            i = i + 1
            # if i == 5:
            #     break
            # test_image = Image.open(image['path']).resize((width, height), Image.ANTIALIAS)
            # test_image = Image.open(image['path']).convert('RGB').resize((self.CONFIG['MODEL_CONFIG']['IMG_HEIGHT'], self.CONFIG['MODEL_CONFIG']['IMG_WIDTH']), Image.ANTIALIAS)
            # test_image = np.array(np.expand_dims(test_image, axis=0), dtype=np.float32)
            test_image = np.array(Image.open(image['path']).resize((self.CONFIG['MODEL_CONFIG']['IMG_HEIGHT'], self.CONFIG['MODEL_CONFIG']['IMG_WIDTH']))).astype(np.float32) / self.CONFIG['MODEL_CONFIG']['IMG_WIDTH'] - 1
            test_image = test_image.reshape(1,self.CONFIG['MODEL_CONFIG']['IMG_HEIGHT'],self.CONFIG['MODEL_CONFIG']['IMG_WIDTH'],3)

            # Initialize the TFLite interpreter
            if interpreter == None:
                interpreter = tf.lite.Interpreter(model_path=self.CONFIG["MODEL_PATH"]+'/model_lighter.tflite')

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # Adjust graph input to handle batch tensor
            interpreter.resize_tensor_input(input_details[0]['index'], test_image.shape) #(batch_size, 512, 512, 3)

            # Adjust output #1 in graph to handle batch tensor
            interpreter.resize_tensor_input(output_details[0]['index'], test_image.shape) #(batch_size, 512, 512, 3)
            interpreter.allocate_tensors()

            # _, height, width, _ = input_details[0]['shape']
            # input_shape = input_details[0]['shape']
            # print(input_shape)
        
            # If required, quantize the input layer (from float to integer)
            input_scale, input_zero_point = input_details[0]["quantization"]
            if (input_scale, input_zero_point) != (0.0, 0):
                test_image = test_image / input_scale + input_zero_point
                test_image = test_image.astype(input_details[0]["dtype"])

            interpreter.set_tensor(input_details[0]["index"], test_image)
            interpreter.invoke()
            # print(output_details[0]) # Shape and shape signature are always (1, 512, 512, 3)        
            pred_vec = interpreter.get_tensor(output_details[0]["index"])
            # print(pred_vec)
            # If required, dequantized the output layer (from integer to float)
            output_scale, output_zero_point = output_details[0]["quantization"]
            if (output_scale, output_zero_point) != (0.0, 0):
                pred_vec = output_data.astype(np.float32)
                pred_vec = (pred_vec - output_zero_point) * output_scale
            
            # output = np.array(output_data)
            # output = np.squeeze(output_data[0])
            # print(output)

            pred_class =[]
            confidence = np.round(pred_vec.max(),2) 
            # confidence = pred_vec.max()
            # print('confidence: >> ', confidence)

            if confidence > 0.4:
                # pc = pred_vec.argmax()
                pc = np.argmax(pred_vec[0])
                pred_class.append( (pc, confidence) )
            else:
                pred_class.append( (0, 0) )
            if pred_class:
                # print('pred_class: >> ', pred_class)
                # print('confidence: >> ', confidence)
                result = self.get_display_string(pred_class, label_dict) 
                # print('Result: >> ', result)
                # print('\n')
            predictions.append({'name': image['name'], 'prediction': pred_class, 'result': result})
        return predictions

        
        # output = np.squeeze(tflite_interpreter_output[0])
        # output = np.argmax(tflite_interpreter_output[0], axis=0)
        # print('probabilities: >>> ', output)
        # if output_details['dtype'] == np.uint8:
        #     scale, zero_point = output_details[0]['quantization']
        #     output = scale * (output - zero_point)

        # score = (np.exp(output).T / np.exp(output).sum(axis=-1)).T
        # print(
        #     "This image most likely belongs to {} with a {:.2f} percent confidence."
        #     .format(label_dict[np.argmax(score)], 100 * np.max(score))
        # )
        
        # If required, quantize the input layer (from float to integer)
        # input_scale, input_zero_point = input_details["quantization"]
        # if (input_scale, input_zero_point) != (0.0, 0):
        #     x_test_ = x_test_ / input_scale + input_zero_point
        #     x_test_ = x_test_.astype(input_details["dtype"])
        
        # # Invoke the interpreter
        # y_pred = np.empty(x_test_.size, dtype=output_details["dtype"])
        # for i in range(len(x_test_)):
        #     interpreter.set_tensor(input_details["index"], [x_test_[i]])
        #     interpreter.invoke()
        #     y_pred[i] = interpreter.get_tensor(output_details["index"])[0]
        
        # # If required, dequantized the output layer (from integer to float)
        # output_scale, output_zero_point = output_details["quantization"]
        # if (output_scale, output_zero_point) != (0.0, 0):
        #     y_pred = y_pred.astype(np.float32)
        #     y_pred = (y_pred - output_zero_point) * output_scale



        # Post-processing: remove batch dimension and find the digit with highest
        # probability.
        # prediction = np.argmax(output()[0])
        # print('prediction: >> ',prediction)
        
       
        
   
       
       
        
  