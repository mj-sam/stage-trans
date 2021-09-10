#!/usr/bin/env python3

def stage_training(model,train_data,train_lbl,
                    validation_data, validation_lbl,
                    callback,epochs = [1000,200],verbose = False,
                    loss = 'categorical_crossentropy',optimizer = 'adam'):
    '''	
        Train a ConvNet tensorflow model using stage training

        Return:	A trained tensorflow Model 
	'''
    model.compile(loss=loss, optimizer=optimizer)
    model.fit(x=train_data, y=train_lbl, 
                batch_size=16, epochs=epochs[0], verbose= False, 
                callbacks=callback,
                validation_data = (validation_data, validation_lbl), 
                shuffle=True)

    for i in range(3,5):
        model.layers[i].trainable = False

    model.fit(x=train_data, y=train_lbl, 
                    batch_size=16, epochs=epochs[1], 
                    verbose= verbose, callbacks=callback,
                    validation_data = (validation_data, validation_lbl), 
                    shuffle=True)

    for i in range(3,5):
        model.layers[i].trainable = True

    for i in range(1,3):
        model.layers[i].trainable = False

    model.fit(x=train_data, y=train_lbl, 
                    batch_size=16, epochs=epochs[1], 
                    verbose= verbose, callbacks=callback,
                    validation_data = (validation_data, validation_lbl), 
                    shuffle=True)

    for i in range(1,3):
        model.layers[i].trainable = True

    model.fit(x=train_data, y=train_lbl, 
                    batch_size=16, epochs=epochs[1],
                    verbose= verbose, callbacks=callback,
                    validation_data = (validation_data, validation_lbl), 
                    shuffle=True)

    return model


def standard_training(
            model,train_data,train_lbl,
            validation_data, validation_lbl,
            callback,epochs = [1000,200],verbose = False,
            loss = 'categorical_crossentropy',optimizer = 'adam'):
    '''	
        Train a ConvNet tensorflow model using standard training

        Return:	A trained tensorflow Model 
	'''
    model.compile(loss, optimizer)
    model.fit(x=train_data, y=train_lbl, 
                batch_size=16, epochs=epochs[0], verbose= False, 
                callbacks=callback,
                validation_data = (validation_data, validation_lbl), 
                shuffle=True)
    return model