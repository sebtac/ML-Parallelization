# REQUIRED PACKAGES
#import os
#os.system("pip install tensorflow")
#os.system("pip install tensorflow-io")
#os.system("pip install sagemaker-tensorflow")

import argparse

# ANALYSIS PARAMETERS
virtual_devices = 0 # to use VIRTUAL CPUs or GPUs; 0-No, 1+ Number of virtual devices (work with 2+)
download_MINIST = 0 # Should we download original MINIST data?
create_TFRECORDS = 0 # ShouLd we create TFRECORD files with MINIST data?
TFRECORDS_shards = [1,4,8,12,24,36,48,60,120] # How many shards of data to create for MINST TFRECORD? [9 different scenarios]
USE_PIPPED = 0 # Should we use AWS PIPING -- Possibly to be setup on AWS Level

# CURRENLY UNSPECIFFICED (i.e. ALL AVAILABLE) -- CODE BELOW NEED CHANGES TO ALLOW RESOURCE SPECIFICATION
devices = "" # ["CPU:0", "CPU:1"], ["GPU:0", "GPU:1", "GPU:2", "GPU:3"], <<NOTHING>> == ALL AVAILABLE

sharding_setup = 0 # To Control Distributed Dataset Sharding Poilicy -- Default AUTO.
data_init_strat = "EDD_MULTIFILES" # "EDD", "DDFF", "EDD_MULLTIFILES", DDFF_MULTIFILES" (experimental_distribute_dataset, distribute_datasets_from_function)

#TF MODEL PARAMETERS
cnn_n = 1024 # BASE_N: 32, 128, 1024
cnn_layers = 3 # 1,2,3,
worker = "GPU" # "GPU", "CPU", "ORIGINAL"
tf_function = 1


BUFFER_SIZE = 60000 #len(train_images)

BATCH_SIZE_PER_REPLICA = 64 # 64, 1024

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument("--input_dist_mode", type=str, default="File")
arg_parser.add_argument("--shards_on_input", type=int, default=4)
arg_parser.add_argument("--epochs", type=int, default=3)
arg_parser.add_argument("--download_raw_data", type=int, default=0) # ShouLd we create TFRECORD files with MINIST data?
arg_parser.add_argument("--train_dist_mode", type=str, default="TF")

args, _ = arg_parser.parse_known_args()

traing_tfrecord_shard = args.shards_on_input
input_dist_mode = args.input_dist_mode
train_dist_mode = args.train_dist_mode
EPOCHS = args.epochs
download_MINIST = args.download_raw_data # Should we download original MINIST data?

data_source = "S3" # S3, Local

# INITIALIZATION
import os, time
import numpy as np

if train_dist_mode == "TF":
    
    import tensorflow as tf

    # TF VIRTUAL DEVICES
    if virtual_devices > 0:
        N_VIRTUAL_DEVICES = virtual_devices
        physical_devices = tf.config.list_physical_devices(worker) # worker: "CPU", "GPU"
        #physical_devices = tf.config.list_physical_devices("GPU")
        tf.config.set_logical_device_configuration(physical_devices[0], 
                                                   [tf.config.LogicalDeviceConfiguration() for _ in range(N_VIRTUAL_DEVICES)])

        print("PHYSICAL DEVICES:", physical_devices)
        print("LOGIAL DEVICES:", tf.config.list_logical_devices(worker)) # worker: "CPU", "GPU"

    print(tf.__version__)

    # MINIST DATASET

    if download_MINIST == 1:
        fashion_mnist = tf.keras.datasets.fashion_mnist

        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

        # Add a dimension to the array -> new shape == (28, 28, 1)
        # This is done because the first layer in our model is a convolutional
        # layer and it requires a 4D input (batch_size, height, width, channels).
        # batch_size dimension will be added later on.
        train_images = train_images[..., None]
        test_images = test_images[..., None]

        # Scale the images to the [0, 1] range.
        train_images = train_images / np.float32(255)
        test_images = test_images / np.float32(255)

    print("DONE!")

    # CRATE DISTRIBUTED TFRECORDs

    if create_TFRECORDS == 1:
        for shard in TFRECORDS_shards:
            if TFRECORDS_shards == 1:
                # SINGLE TFRECORD FILE -- USED MAINLY FOR TEST/VALIDATION FILE
                file_paths = ['MNIST_test_data.tfrecords', 'MNIST_train_data.tfrecords']
                for file_path in file_paths:

                    with tf.io.TFRecordWriter(file_path) as writer:
                        for i in range(test_images.shape[0]):
                            image = test_images[i]
                            label = test_labels[i]
                            serialized_image = tf.io.serialize_tensor(image)
                            serialized_label = tf.io.serialize_tensor(label)
                            features = {'images': tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_image.numpy()])),
                                       'labels': tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_label.numpy()]))}
                            example_message = tf.train.Example(features=tf.train.Features(feature=features))

                            writer.write(example_message.SerializeToString())

            elif TFRECORDS_shards > 1:
                # MULTIPLE TFRECORD FILES - TRAIN FILES

                files_number = TFRECORDS_shards

                image_dict = {}
                label_dict = {}

                for i in range(files_number):
                    image_dict[i] = []
                    label_dict[i] = []

                for i in range(train_images.shape[0]):
                    image = train_images[i]
                    label = train_labels[i]

                    file_id = np.random.randint(0,files_number)

                    image_dict[file_id].append(image)
                    label_dict[file_id].append(label)

                for key in image_dict.keys():

                    file_path = './TFRecords/MNIST_train_data_{}_{}.tfrecords'.format(files_number, key)
                    print(file_path)

                    with tf.io.TFRecordWriter(file_path) as writer:
                        for i in range(len(image_dict[key])):
                            image = train_images[i]
                            label = train_labels[i]
                            serialized_image = tf.io.serialize_tensor(image)
                            serialized_label = tf.io.serialize_tensor(label)
                            features = {'images': tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_image.numpy()])),
                                       'labels': tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_label.numpy()]))}
                            example_message = tf.train.Example(features=tf.train.Features(feature=features))

                            writer.write(example_message.SerializeToString())

    print("DONE!")

    # SET TF DISTRIBUTION STRATEGY

    # If the list of devices is not specified in
    # `tf.distribute.MirroredStrategy` constructor, they will be auto-detected.
    strategy = tf.distribute.MirroredStrategy()

    # SPECIFY INDIVIDUAL GPUs (CPUs):
    # - IF NOT SPECIFIED THAN IT USES ALL AVAILABLE RESOURCES; DEFAULT = "GPU"
    # - TF DOES NOT RECOGNIZE INDIVIDUAL CPU CORES THUS IT RUNS ONLY ON "CPU: 0"

    #strategy = tf.distribute.MirroredStrategy(devices = ["CPU:0"])
    #strategy = tf.distribute.MirroredStrategy(devices = ["CPU:0", "CPU:1"])
    #strategy = tf.distribute.MirroredStrategy(devices = ["GPU:0"]) #
    #strategy = tf.distribute.MirroredStrategy(devices = ["GPU:0", "GPU:1"])

    num_replicas = strategy.num_replicas_in_sync

    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * num_replicas
    LR = 0.001 * num_replicas

    print('Number of replicas: {}'.format(num_replicas))
    print("GLOBAL_BATCH_SIZE", GLOBAL_BATCH_SIZE)
    print('Learning Rate: {}'.format(LR))

    files_number = traing_tfrecord_shard

    SM_CHANNEL_TRAIN = os.environ["SM_CHANNEL_TRAIN"]
    print("SM_CHANNEL_TRAIN", SM_CHANNEL_TRAIN)

    if input_dist_mode == "Pipe":

        from sagemaker_tensorflow import PipeModeDataset
        train_dataset = PipeModeDataset(channel="train", record_format='TFRecord')

    elif input_dist_mode in ["FastFile", "File"]:

        file_names = tf.data.Dataset.list_files(SM_CHANNEL_TRAIN + "/MNIST_train_data_{}_*.tfrecords".format(files_number))

        train_dataset = file_names.interleave(tf.data.TFRecordDataset,
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                        cycle_length=None,
                                        block_length=None)
                                        #cycle_length=BATCH_SIZE_PER_REPLICA,
                                        #block_length=BATCH_SIZE_PER_REPLICA)    

        print("Found Files:", file_names.cardinality().numpy())
        print("File Names:", file_names)

    print("train_dataset", train_dataset)


    """
    # LIST OBJECTS IN S3 BUCKET (OPTION: FILTERED WITH STRING PATTERN)

    import boto3
    s3_client = boto3.client('s3')

    my_buckets = s3_client.list_buckets()
    #print(my_buckets)

    #objects = s3_client.list_objects_v2(Bucket="mnist-tdrecords/train/{}".format(hyperparameters["shards_on_input"]))
    objects = s3_client.list_objects_v2(Bucket="mnist-tdrecords")

    #print("objects['Contents']", objects['Contents'])

    file_names = []
    for obj in objects['Contents']:
        if "train/{}".format(args.shards_on_input) in obj['Key'] and ".tfrecords" in obj['Key']:
            file_names.append(obj['Key'])
            #print(obj['Key'])

    print("file_names", file_names)
    """

    """
    import os
    files_in_SM_CHANNEL_TRAIN = os.listdir(SM_CHANNEL_TRAIN)
    print("files_in_SM_CHANNEL_TRAIN", files_in_SM_CHANNEL_TRAIN)
    """

    # CREATE DISTRIBUTED DATASET

    # SHARDING MUST BE SPECIFIED BEFORE CREATION OF THE DISTRIBUTED DATASET BUT BY DEFAULT IT USES AUTO POLICY
    if sharding_setup == 1:
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.AUTO # FIRST TRY: FILE THEN DATA
        #options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE #ASSIGNS FILES TO WORKERS -- number of FILES>number of WORKERS, File sizes fairly balanced
        #options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA #PRELOAD ALL DATA to EACH WORKER?!?!?! but USES ONLY GIVEN SHARD DATA PER WORKER ?!?!?
        #options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF #EACH WORKER GETS AND PROCESS ALL DATA!
        train_dataset = train_dataset.with_options(options) # .with_options() WORKS ONLY WITH THE ORIGINAL DATASET NOT THE DISTRIBUTED ONE!!!

    if data_init_strat == "EDD":
        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE) # OPTIONALY ADD .repeat(EPOCHS)
        test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(GLOBAL_BATCH_SIZE)

        with strategy.scope(): # NEED TO CREATE DISTRIBUTED DATASET WITHIN THE SCOPE -- IT SPED UP TRAINIG 3x

            train_dataset_dist = strategy.experimental_distribute_dataset(train_dataset)
            test_dataset_dist = strategy.experimental_distribute_dataset(test_dataset)

            # PREFETCHING
            # By default, .experimental_distribute_dataset() adds a prefetch transformation at the end of the user provided tf.data.Dataset instance.
            # The argument to the prefetch transformation which is buffer_size is equal to the number of replicas in sync.

    elif data_init_strat == "DDFF": # BETTER PERFORMANCE WITH MULTIWORKER DISTRIBUTED TRAINING
        with strategy.scope(): # NEED TO CREATE DISTRIBUTED DATASET WITHIN THE SCOPE -- IT SPED UP TRAINIG 3x
            def dataset_train_fn(input_context):
                #dataset = dataset_train
                dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE)
                dataset = dataset.interleave(num_parallel_calls=tf.data.AUTOTUNE) # in case we work with miltipel files this proceses multipe files in parallel
                dataset = dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
                #batch_size = input_context.get_per_replica_batch_size(GLOBAL_BATCH_SIZE) # alternative way of getting it
                dataset = dataset.batch(BATCH_SIZE_PER_REPLICA) # HERE IT MUST BE BATCHED WITH PER REPLICA BS
                #dataset = dataset.map(scale,, num_parallel_calls=tf.data.AUTOTUNE).cache().shuffle(BUFFER_SIZE).repeat(EPOCHS) ### HEAVING MAP AFTER BATCH SHOUDL DO MAPPIN ON BATCHES NOT NDIVIDUAL EXAMPLES
                # .distribute_datasets_from_function() DOES NOT ADD PREFETCH AUTOMATICALL -- ADD IT MANUALY
                #dataset = dataset.prefetch(2)  # This prefetches 2 batches per device.
                #dataset = dataset.prefetch(strategy.num_replicas_in_sync) # This might be an overkil with multipel DEVICES.
                dataset = dataset.prefetch(tf.data.AUTOTUNE) # NOT TESTED IF BEST!!!

                return dataset

            train_dataset_dist = strategy.distribute_datasets_from_function(dataset_train_fn)

            def dataset_test_fn(input_context):
                #dataset = dataset_train\
                dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).shuffle(BUFFER_SIZE)
                dataset = dataset.interleave(num_parallel_calls=tf.data.AUTOTUNE) # in case we work with miltipel files this proceses multipe files in parallel
                dataset = dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
                #batch_size = input_context.get_per_replica_batch_size(GLOBAL_BATCH_SIZE) # alternative way of getting it
                dataset = dataset.batch(BATCH_SIZE_PER_REPLICA) # HERE IT MUST BE BATCHED WITH PER REPLICA BS
                #dataset = dataset.map(scale, num_parallel_calls=tf.data.AUTOTUNE).cache().shuffle(BUFFER_SIZE).repeat(EPOCHS) ### HEAVING MAP AFTER BATCH SHOUDL DO MAPPIN ON BATCHES NOT NDIVIDUAL EXAMPLES
                # .distribute_datasets_from_function() DOES NOT ADD PREFETCH AUTOMATICALL -- ADD IT MANUALY
                #dataset = dataset.prefetch(2)  # This prefetches 2 batches per device.
                #dataset = dataset.prefetch(strategy.num_replicas_in_sync) # This might be an overkil with multipel DEVICES.
                dataset = dataset.prefetch(tf.data.AUTOTUNE) # NOT TESTED IF BEST!!!
                return dataset

            test_dataset_dist = strategy.distribute_datasets_from_function(dataset_test_fn)


    # Read TFRecord file
    def _parse_tfr_element(element):
        parse_dic = {'images': tf.io.FixedLenFeature([], tf.string),
                    'labels': tf.io.FixedLenFeature([], tf.string),} # Note that it is tf.string, not tf.float32

        example_message = tf.io.parse_single_example(element, parse_dic)

        images = example_message['images'] # get byte string
        labels = example_message['labels'] # get byte string

        image_feature = tf.io.parse_tensor(images, out_type=tf.float32) # restore 2D array from byte string
        label_feature = tf.io.parse_tensor(labels, out_type=tf.uint8)

        # SET SHAPES AS tf.io.parse_single_example() DOES NOT READ THEM AUTOMATICALLY AND THIS BREAKS THE DISTIRBUTED DATASET GENERATION
        image_feature.set_shape((28, 28, 1)) # PROVIDE ALL DIMS BUT THE BATCH ONE!
        label_feature.set_shape(()) # PROVIDE ALL DIMS BUT THE BATCH ONE!

        return (image_feature, label_feature)

    if data_init_strat == "EDD_MULTIFILES":

        files_number = traing_tfrecord_shard

        #file_names = tf.data.Dataset.list_files("./TFRecords/MNIST_train_data_{}_*.tfrecords".format(files_number))
        #file_names = tf.data.Dataset.list_files(f"s3://mnist-tdrecords/train/4")

        #print("Found Files:", file_names.cardinality().numpy())

        #for file_name in file_names:
        #    print(file_name.numpy().decode("utf-8"))
        """
        train_dataset = file_names.interleave(tf.data.TFRecordDataset,
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                        cycle_length=None,
                                        block_length=None)
                                        #cycle_length=BATCH_SIZE_PER_REPLICA,
                                        #block_length=BATCH_SIZE_PER_REPLICA)
        """
        #print("train_dataset", train_dataset)
        dataset = train_dataset.map(_parse_tfr_element)
        #print("dataset 1", dataset)
        dataset = dataset.shuffle(BUFFER_SIZE)
        #print("dataset 2", dataset)
        dataset = dataset.batch(GLOBAL_BATCH_SIZE)
        print("dataset 3", dataset)

        # EXPECTED DATASET SHAPES & TYPES
        #<BatchDataset shapes: ((None, 28, 28, 1), (None,)), types: (tf.float32, tf.uint8)>

        #train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE) # OPTIONALY ADD .repeat(EPOCHS)

        if input_dist_mode != "File" and download_MINIST == 1:
            test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(GLOBAL_BATCH_SIZE)

        with strategy.scope(): # NEED TO CREATE DISTRIBUTED DATASET WITHIN THE SCOPE -- IT SPED UP TRAINIG 3x
            #train_dataset_dist = strategy.experimental_distribute_dataset(train_dataset)
            train_dataset_dist = strategy.experimental_distribute_dataset(dataset)
        if input_dist_mode != "File" and download_MINIST == 1:
            test_dataset_dist = strategy.experimental_distribute_dataset(test_dataset)

            # PREFETCHING
            # By default, .experimental_distribute_dataset() adds a prefetch transformation at the end of the user provided tf.data.Dataset instance.
            # The argument to the prefetch transformation which is buffer_size is equal to the number of replicas in sync.

    elif data_init_strat == "DDFF_MULTIFILES": # BETTER PERFORMANCE WITH MULTIWORKER DISTRIBUTED TRAINING

        files_number = traing_tfrecord_shard

        #file_names = tf.data.Dataset.list_files("./TFRecords/MNIST_train_data_{}_*.tfrecords".format(files_number))

        print("Found Files:", file_names.cardinality().numpy())

        #for file_name in file_names:
        #    print(file_name.numpy().decode("utf-8"))

        train_dataset = file_names.interleave(tf.data.TFRecordDataset,
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                        cycle_length=None,
                                        block_length=None)
                                        #cycle_length=BATCH_SIZE_PER_REPLICA,
                                        #block_length=BATCH_SIZE_PER_REPLICA)

        #print("train_dataset", train_dataset)
        dataset = train_dataset.map(_parse_tfr_element)
        #print("dataset 1", dataset)
        dataset = dataset.shuffle(BUFFER_SIZE)
        #print("dataset 2", dataset)
        dataset = dataset.batch(GLOBAL_BATCH_SIZE)
        print("dataset 3", dataset)

        # EXPECTED DATASET SHAPES & TYPES
        #<BatchDataset shapes: ((None, 28, 28, 1), (None,)), types: (tf.float32, tf.uint8)>

        with strategy.scope(): # NEED TO CREATE DISTRIBUTED DATASET WITHIN THE SCOPE -- IT SPED UP TRAINIG 3x

            def dataset_train_fn(input_context):
                global dataset # WHY DO WE NEED TO SPECIFY IT AND FOR test_images, test_labels NOT???
                dataset = dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
                dataset = dataset.batch(BATCH_SIZE_PER_REPLICA) # HERE IT MUST BE BATCHED WITH PER REPLICA BS
                dataset = dataset.prefetch(tf.data.AUTOTUNE) # NOT TESTED IF BEST!!!

                return dataset

            train_dataset_dist = strategy.distribute_datasets_from_function(dataset_train_fn)

            def dataset_test_fn(input_context):
                dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).shuffle(BUFFER_SIZE)
                dataset = dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
                dataset = dataset.batch(BATCH_SIZE_PER_REPLICA) # HERE IT MUST BE BATCHED WITH PER REPLICA BS
                dataset = dataset.prefetch(tf.data.AUTOTUNE) # NOT TESTED IF BEST!!!

                return dataset

            test_dataset_dist = strategy.distribute_datasets_from_function(dataset_test_fn)

    else:
        # ORIGINAL CODE -- NO DISTRIBUTED DATASET
        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)
        test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(GLOBAL_BATCH_SIZE)

        # COMMENTED OUT TO IMITATE NO DISTRIBUTED DATASET
        #train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
        #test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)

    # BUILD MODEL & TF OBJECTS

    if worker == "GPU":
        def create_model():
            model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(28, 28)),
                tf.keras.layers.Reshape(target_shape=( 28, 28, 1)),
                #if cnn_layers >= 1:
                tf.keras.layers.Conv2D(cnn_n, 3, activation='relu',data_format='channels_last'),
                tf.keras.layers.MaxPooling2D(),
                #if cnn_layers >= 2:
                tf.keras.layers.Conv2D(cnn_n*2, 3, activation='relu',data_format='channels_last'),
                tf.keras.layers.MaxPooling2D(),
                #if cnn_layers >= 3:
                tf.keras.layers.Conv2D(cnn_n*4, 3, activation='relu',data_format='channels_last'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(10)])

            return model

    if worker == "CPU": #MAXPOOLING DOES NOT WORK ON CPU IN SOME CONFIGURATIONS
        def create_model():
            model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(28, 28)),
                tf.keras.layers.Reshape(target_shape=( 28, 28, 1)),
                #if cnn_layers >= 1:
                tf.keras.layers.Conv2D(cnn_n, 3, activation='relu',data_format='channels_last'),
                #tf.keras.layers.MaxPooling2D(),
                #if cnn_layers >= 2:
                tf.keras.layers.Conv2D(cnn_n*2, 3, activation='relu',data_format='channels_last'),
                #tf.keras.layers.MaxPooling2D(),
                #if cnn_layers >= 3:
                tf.keras.layers.Conv2D(cnn_n*4, 3, activation='relu',data_format='channels_last'),
                #tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(10)])

            return model

    if worker == "ORIGINAL": # CPU, AS IN THE ORIGINAL EXAMPLE
        def create_model():
            model = tf.keras.Sequential([
                #tf.keras.layers.InputLayer(input_shape=(28, 28)),
                #tf.keras.layers.Reshape(target_shape=( 28, 28, 1)),
                tf.keras.layers.Conv2D(32, 3, activation='relu',data_format='channels_last'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(64, 3, activation='relu',data_format='channels_last'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(10)])

            return model


    # Create a checkpoint directory to store the checkpoints.
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    with strategy.scope():
        # Set reduction to `NONE` so you can do the reduction afterwards and divide by
        # global batch size.
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE)

        def compute_loss(labels, predictions, model_losses):
            per_example_loss = loss_object(labels, predictions)
            loss = tf.nn.compute_average_loss(per_example_loss,
                                              global_batch_size=GLOBAL_BATCH_SIZE)
            if model_losses:
                loss += tf.nn.scale_regularization_loss(tf.add_n(model_losses))
            return loss

        test_loss = tf.keras.metrics.Mean(name='test_loss')

        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        # A model, an optimizer, and a checkpoint must be created under `strategy.scope`.

        model = create_model()

        optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    def train_step(inputs):
        images, labels = inputs

        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = compute_loss(labels, predictions, model.losses)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_accuracy.update_state(labels, predictions)

        return loss

    def test_step(inputs):
        images, labels = inputs

        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)

        test_loss.update_state(t_loss)
        test_accuracy.update_state(labels, predictions)

    # RUN TRAINING WITHOUT @tf.function (ON TRAIN FUNCTION)

    if tf_function == 0:

        @tf.function
        def distributed_train_step(dataset_inputs):
            per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        @tf.function
        def distributed_test_step(dataset_inputs):
            return strategy.run(test_step, args=(dataset_inputs,))

        # RESTART MODEL - this adds up to 5 secounds to the executiontime as the model must be initialized! -- FIND BETTER WAY
        #"""
        with strategy.scope():
            # Set reduction to `NONE` so you can do the reduction afterwards and divide by
            # global batch size.
            loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                        reduction=tf.keras.losses.Reduction.NONE)

            def compute_loss(labels, predictions, model_losses):
                per_example_loss = loss_object(labels, predictions)
                loss = tf.nn.compute_average_loss(per_example_loss,
                                                  global_batch_size=GLOBAL_BATCH_SIZE)
                if model_losses:
                    loss += tf.nn.scale_regularization_loss(tf.add_n(model_losses))
                return loss

            test_loss = tf.keras.metrics.Mean(name='test_loss')

            train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
            test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

            model = create_model()
            optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
            checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        #"""

        start_time = time.time()
        for epoch in range(EPOCHS):
            # TRAIN LOOP
            total_loss = 0.0
            num_batches = 0
            for x in train_dataset_dist:
                total_loss += distributed_train_step(x)
                num_batches += 1
            #print("num_batches", num_batches)
            train_loss = total_loss / num_batches

            # TEST LOOP
            for x in test_dataset_dist:
                distributed_test_step(x)

            if epoch % 2 == 0:
                checkpoint.save(checkpoint_prefix)

            template = ("Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}")
            print(template.format(epoch + 1, train_loss,
                                  train_accuracy.result() * 100, test_loss.result(),
                                  test_accuracy.result() * 100))

            test_loss.reset_states()
            train_accuracy.reset_states()
            test_accuracy.reset_states()

        print("TIME", time.time() - start_time)


    # RUN TRAINING WITH @tf.function (ON TRAIN FUNCTION)

    if tf_function == 1:

        def test_step(inputs):
            images, labels = inputs

            predictions = model(images, training=False)
            t_loss = loss_object(labels, predictions)

            test_loss.update_state(t_loss)
            test_accuracy.update_state(labels, predictions)

        @tf.function
        def distributed_test_step(dataset_inputs):
            return strategy.run(test_step, args=(dataset_inputs,))        

        # RESTART MODEL - this adds up to 5 secounds to the executiontime as the model must be initialized! -- FIND BETTER WAY

        #"""
        with strategy.scope():
            # Set reduction to `NONE` so you can do the reduction afterwards and divide by
            # global batch size.
            loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True,
                reduction=tf.keras.losses.Reduction.NONE)

            def compute_loss(labels, predictions, model_losses):
                per_example_loss = loss_object(labels, predictions)
                loss = tf.nn.compute_average_loss(per_example_loss,
                                                  global_batch_size=GLOBAL_BATCH_SIZE)
                if model_losses:
                    loss += tf.nn.scale_regularization_loss(tf.add_n(model_losses))
                return loss

            test_loss = tf.keras.metrics.Mean(name='test_loss')

            callback = tf.keras.callbacks.EarlyStopping(monitor='test_accuracy',
                                                        min_delta=0,
                                                        patience=3,
                                                        verbose=0,
                                                        mode='auto',
                                                        baseline=None,
                                                        restore_best_weights=True)
                                                        #start_from_epoch=0) # UNRECOGNIZED ASRGUMENT IN THIS VERSION OF TF

            train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
            test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

            model = create_model()
            optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
            checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        #"""

        @tf.function
        def distributed_train_step(dataset):
            total_loss = 0.0
            num_batches = 0
            for x in dataset:
                #print(x)
                per_replica_losses = strategy.run(train_step, args=(x,))
                total_loss += strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
                num_batches += 1
            #print(num_batches)
            return total_loss / tf.cast(num_batches, dtype=tf.float32), num_batches

        start_time = time.time()

        patience_counter = 0
        test_accuracy_results = [0]
        patience = 3

        for epoch in range(EPOCHS):
            if patience_counter < patience:
                train_loss, num_batches = distributed_train_step(train_dataset_dist)

                # TEST LOOP
                for x in test_dataset_dist:
                    distributed_test_step(x)        

                template = ("Epoch {}, Loss: {}, Train Accuracy: {}, Test Accuracy: {}")
                print(template.format(epoch + 1, train_loss, train_accuracy.result() * 100, test_accuracy.result() * 100))

                #print(test_accuracy.result().numpy(), max(test_accuracy_results), test_accuracy.result().numpy() > max(test_accuracy_results), type(test_accuracy.result().numpy()), type(max(test_accuracy_results)))
                if test_accuracy.result().numpy() *100 > max(test_accuracy_results):
                    #print("APPEND")
                    test_accuracy_results.append(test_accuracy.result().numpy() * 100)
                    patience_counter = 0
                else:
                    #print("PASS")
                    patience_counter += 1

                #print("patience_counter", patience_counter, max(test_accuracy_results), test_accuracy_results)
            else:
                print("Training finalized with best Test Accuracy of " + str(max(test_accuracy_results)))

            train_accuracy.reset_states()
            test_accuracy.reset_states()

        print("TIME", time.time() - start_time)
        print("num_batches", num_batches)

elif train_dist_mode == "SMD":
    
    import tensorflow as tf
    
    # Import SMDataParallel TensorFlow2 Modules
    import smdistributed.dataparallel.tensorflow as dist
    
    # SMDataParallel: Initialize
    dist.init()

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        # SMDataParallel: Pin GPUs to a single SMDataParallel process [use SMDataParallel local_rank() API]
        tf.config.experimental.set_visible_devices(gpus[dist.local_rank()], "GPU")

    # MINIST DATASET
    if dist.rank() == 0 or dist.rank() == 8:
        if download_MINIST == 1:
            fashion_mnist = tf.keras.datasets.fashion_mnist

            (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

            # Add a dimension to the array -> new shape == (28, 28, 1)
            # This is done because the first layer in our model is a convolutional
            # layer and it requires a 4D input (batch_size, height, width, channels).
            # batch_size dimension will be added later on.
            train_images = train_images[..., None]
            test_images = test_images[..., None]

            # Scale the images to the [0, 1] range.
            train_images = train_images / np.float32(255)
            test_images = test_images / np.float32(255)           

    files_number = traing_tfrecord_shard

    SM_CHANNEL_TRAIN = os.environ["SM_CHANNEL_TRAIN"]
    print("SM_CHANNEL_TRAIN", SM_CHANNEL_TRAIN)

    if input_dist_mode == "Pipe":

        from sagemaker_tensorflow import PipeModeDataset
        train_dataset = PipeModeDataset(channel="train", record_format='TFRecord')

    elif input_dist_mode in ["FastFile", "File"]:

        file_names = tf.data.Dataset.list_files(SM_CHANNEL_TRAIN + "/MNIST_train_data_{}_*.tfrecords".format(files_number))

        train_dataset = file_names.interleave(tf.data.TFRecordDataset,
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                        cycle_length=None,
                                        block_length=None)
                                        #cycle_length=BATCH_SIZE_PER_REPLICA,
                                        #block_length=BATCH_SIZE_PER_REPLICA)    

        print("Found Files:", file_names.cardinality().numpy())
        print("File Names:", file_names)

    print("Replica", dist.rank(), "train_dataset", train_dataset)

    num_replicas = dist.size()

    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * num_replicas
    LR = 0.001 * num_replicas

    print("Replica", dist.rank(), 'Number of replicas: {}'.format(num_replicas))
    print("Replica", dist.rank(), "GLOBAL_BATCH_SIZE", GLOBAL_BATCH_SIZE)
    print("Replica", dist.rank(), 'Learning Rate: {}'.format(LR))

    # Read TFRecord file
    def _parse_tfr_element(element):
        parse_dic = {'images': tf.io.FixedLenFeature([], tf.string),
                    'labels': tf.io.FixedLenFeature([], tf.string),} # Note that it is tf.string, not tf.float32

        example_message = tf.io.parse_single_example(element, parse_dic)

        images = example_message['images'] # get byte string
        labels = example_message['labels'] # get byte string

        image_feature = tf.io.parse_tensor(images, out_type=tf.float32) # restore 2D array from byte string
        label_feature = tf.io.parse_tensor(labels, out_type=tf.uint8)

        # SET SHAPES AS tf.io.parse_single_example() DOES NOT READ THEM AUTOMATICALLY AND THIS BREAKS THE DISTIRBUTED DATASET GENERATION
        image_feature.set_shape((28, 28, 1)) # PROVIDE ALL DIMS BUT THE BATCH ONE!
        label_feature.set_shape(()) # PROVIDE ALL DIMS BUT THE BATCH ONE!

        return (image_feature, label_feature)

    #print("train_dataset", train_dataset)
    dataset = train_dataset.map(_parse_tfr_element)
    #print("dataset 1", dataset)
    dataset = dataset.shuffle(BUFFER_SIZE)
    #print("dataset 2", dataset)
    dataset = dataset.batch(GLOBAL_BATCH_SIZE)
    print("dataset 3", dataset)
                
    if worker == "GPU":
        def create_model():
            model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(28, 28)),
                tf.keras.layers.Reshape(target_shape=( 28, 28, 1)),
                #if cnn_layers >= 1:
                tf.keras.layers.Conv2D(cnn_n, 3, activation='relu',data_format='channels_last'),
                tf.keras.layers.MaxPooling2D(),
                #if cnn_layers >= 2:
                tf.keras.layers.Conv2D(cnn_n*2, 3, activation='relu',data_format='channels_last'),
                tf.keras.layers.MaxPooling2D(),
                #if cnn_layers >= 3:
                tf.keras.layers.Conv2D(cnn_n*4, 3, activation='relu',data_format='channels_last'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(10)])

            return model

    model = create_model()     
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)               

    # Set reduction to `NONE` so you can do the reduction afterwards and divide by
    # global batch size.
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) #, reduction=tf.keras.losses.Reduction.NONE)

    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    @tf.function
    def training_step(images, labels, first_batch):
        with tf.GradientTape() as tape:
            probs = model(images, training=True)
            loss_value = loss_object(labels, probs)

        # SMDataParallel: Wrap tf.GradientTape with SMDataParallel's DistributedGradientTape
        tape = dist.DistributedGradientTape(tape)

        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if first_batch:
            # SMDataParallel: Broadcast model and optimizer variables
            dist.broadcast_variables(model.variables, root_rank=0)
            dist.broadcast_variables(optimizer.variables(), root_rank=0)

        # SMDataParallel: all_reduce call
        loss_value = dist.oob_allreduce(loss_value)  # Average the loss across workers
        return loss_value

    epoch = 0
    batch_id = 0
    
    import time

    @tf.function
    def test_accuracy_step(inputs, labels):
        predictions = model(images, training=False)
        test_accuracy.update_state(labels, predictions)

    @tf.function
    def train_accuracy_step(inputs, labels):
        predictions = model(images, training=False)
        train_accuracy.update_state(labels, predictions)
        
    start_time = time.time()
    while epoch < EPOCHS:
        for (images, labels) in dataset:
            #for batch, (images, labels) in enumerate(dataset.take(10000 // dist.size())):
            loss_value = training_step(images, labels, batch_id == 0)

            if batch_id % 50 == 0 and dist.rank() == 0:
                print("Step #%d\tLoss: %.6f" % (batch_id, loss_value))
            batch_id += 1

        if dist.rank() == 0:
            
            # ERROR WITH SHAPES DUE TO MERGE OF CODE FROM TWO DIFFERENT EXAMPLES -- TO BE WORKED OUT!!!
            #test_accuracy_step(test_images, test_labels) 
            
            # FOR SMD train_accuracy.result() ARE OFF COMPARED WITH OTHER APPROACHES, EITHER DUE TO BATCH SIZE DIFFERNCES OR FROM LOGITS TRAINING -- WORK IT OUT!!!

            # ERROR WITH SHAPES DUE TO MERGE OF CODE FROM TWO DIFFERENT EXAMPLES -- TO BE WORKED OUT!!!            
            #train_accuracy_step(train_images, train_labels) 
            
            template = ("Epoch {}, Loss: {}, Train Accuracy: {}, Test Accuracy: {}")
            print(template.format(epoch +1, loss_value, train_accuracy.result() * 100, test_accuracy.result() * 100))
            
            train_accuracy.reset_states()
            test_accuracy.reset_states()            
            
        epoch += 1

    if dist.rank() == 0:        
        print("Training Time", time.time() - start_time)
        
    # SMDataParallel: Save checkpoints only from master node.
    #if dist.rank() == 0:
    #    model.save(os.path.join(checkpoint_dir, "1"))

    
elif train_dist_mode == "MPI": # @AWS SaagMaker: MPI==HOROVOD!!!
    
    # Change 1: Import horovod and keras backend    
    import horovod.tensorflow.keras as hvd
    #import tensorflow.keras.backend as K
    import tensorflow.compat.v1.keras.backend as K
    
    # Change 2: Initialize horovod and get the size of the cluster
    hvd.init()
    
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(hvd.local_rank())

    import tensorflow as tf

    size = hvd.size()
    
    physical_devices = tf.config.list_physical_devices('GPU')
    print(hvd.rank(), "Num GPUs:", len(physical_devices), physical_devices)

    # Change 3 - Pin GPU to local process (one GPU per process)
    
    #config = tf.ConfigProto()
    print("hvd.rank", hvd.rank(), hvd.local_rank())
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(0)
    #config.gpu_options.visible_device_list = str(hvd.local_rank())
    K.set_session(tf.compat.v1.Session(config=config))
    
    # MINIST DATASET
    if hvd.rank() == 0 or hvd.rank() == 8:
        if download_MINIST == 1:
            fashion_mnist = tf.keras.datasets.fashion_mnist

            (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

            # Add a dimension to the array -> new shape == (28, 28, 1)
            # This is done because the first layer in our model is a convolutional
            # layer and it requires a 4D input (batch_size, height, width, channels).
            # batch_size dimension will be added later on.
            train_images = train_images[..., None]
            test_images = test_images[..., None]

            # Scale the images to the [0, 1] range.
            train_images = train_images / np.float32(255)
            test_images = test_images / np.float32(255)
            
            NUM_TRAIN_IMAGES = train_images.shape[0]
            NUM_TEST_IMAGES = test_images.shape[0]
            
            print("SHAPES", train_images.shape, train_labels.shape)

    #NUM_TRAIN_IMAGES = train_images.shape[0]
    #NUM_TEST_IMAGES = test_images.shape[0]

    NUM_TRAIN_IMAGES = 60000
    NUM_TEST_IMAGES = 10000
    
    files_number = traing_tfrecord_shard

    SM_CHANNEL_TRAIN = os.environ["SM_CHANNEL_TRAIN"]
    print("SM_CHANNEL_TRAIN", SM_CHANNEL_TRAIN)

    if input_dist_mode == "Pipe":

        from sagemaker_tensorflow import PipeModeDataset
        train_dataset = PipeModeDataset(channel="train", record_format='TFRecord')

    elif input_dist_mode in ["FastFile", "File"]:
        
        # FOR EVERY HRV.RANK() !!!
        
        #print("hvd.rank()", hvd.rank(), "BEFORE")
        #print("FOLDER:", SM_CHANNEL_TRAIN + "/MNIST_train_data_{}_*.tfrecords".format(files_number))
        
        #import os
        #print("FILES:", os.listdir(SM_CHANNEL_TRAIN))
        
        file_names = tf.data.Dataset.list_files(SM_CHANNEL_TRAIN + "/MNIST_train_data_{}_*.tfrecords".format(files_number))
        #print("hvd.rank()", hvd.rank(), "file_names", file_names)
        #print("hvd.rank()", hvd.rank(), "file_names.numpy()", file_names.numpy())        
        
        train_dataset = file_names.interleave(tf.data.TFRecordDataset,
                                              num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                              cycle_length=None,
                                              block_length=None)
                                              #cycle_length=BATCH_SIZE_PER_REPLICA,
                                              #block_length=BATCH_SIZE_PER_REPLICA)

        print("Found Files:", file_names.cardinality().numpy())
        print("File Names:", file_names)

    num_replicas = size

    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * num_replicas
    LR = 0.001 * num_replicas

    print("Replica", hvd.rank(), 'Number of replicas: {}'.format(num_replicas))
    print("Replica", hvd.rank(), "GLOBAL_BATCH_SIZE", GLOBAL_BATCH_SIZE)
    print("Replica", hvd.rank(), 'Learning Rate: {}'.format(LR))

    # Read TFRecord file
    def _parse_tfr_element(element):
        parse_dic = {'images': tf.io.FixedLenFeature([], tf.string),
                     'labels': tf.io.FixedLenFeature([], tf.string),} # Note that it is tf.string, not tf.float32

        example_message = tf.io.parse_single_example(element, parse_dic)

        images = example_message['images'] # get byte string
        labels = example_message['labels'] # get byte string

        image_feature = tf.io.parse_tensor(images, out_type=tf.float32) # restore 2D array from byte string
        label_feature = tf.io.parse_tensor(labels, out_type=tf.uint8)

        # SET SHAPES AS tf.io.parse_single_example() DOES NOT READ THEM AUTOMATICALLY AND THIS BREAKS THE DISTIRBUTED DATASET GENERATION
        image_feature.set_shape((28, 28, 1)) # PROVIDE ALL DIMS BUT THE BATCH ONE!
        label_feature.set_shape(()) # PROVIDE ALL DIMS BUT THE BATCH ONE!

        return (image_feature, label_feature)

    #print("train_dataset", train_dataset)
    dataset = train_dataset.map(_parse_tfr_element)
    #print("dataset 1", dataset)
    dataset = dataset.shuffle(BUFFER_SIZE)
    #print("dataset 2", dataset)
    dataset = dataset.batch(GLOBAL_BATCH_SIZE)
    print("dataset 3", dataset)    

    if worker == "GPU":
        def create_model():
            model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(28, 28)),
                tf.keras.layers.Reshape(target_shape=( 28, 28, 1)),
                #if cnn_layers >= 1:
                tf.keras.layers.Conv2D(cnn_n, 3, activation='relu',data_format='channels_last'),
                tf.keras.layers.MaxPooling2D(),
                #if cnn_layers >= 2:
                tf.keras.layers.Conv2D(cnn_n*2, 3, activation='relu',data_format='channels_last'),
                tf.keras.layers.MaxPooling2D(),
                #if cnn_layers >= 3:
                tf.keras.layers.Conv2D(cnn_n*4, 3, activation='relu',data_format='channels_last'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(10)])

            return model

    model = create_model()

    # Change 4: Scale the learning using the size of the cluster (total number of workers)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    #checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)               

    # Set reduction to `NONE` so you can do the reduction afterwards and divide by
    # global batch size.
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    #train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    #test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    # Change 5: Wrap your Keras optimizer using Horovod to make it a distributed optimizer
    optimizer = hvd.DistributedOptimizer(optimizer)
    """
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    """
    
    model.compile(loss=loss_object,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    # Change 6: Add callbacks for syncing initial state, and saving checkpoints only on 1st worker (rank 0)
    callbacks = []
    callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
    callbacks.append(hvd.callbacks.MetricAverageCallback())
    #callbacks.append(hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1))
    #callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1))
    
    if hvd.rank() == 0:
        pass
        #callbacks.append(ModelCheckpoint(args.output_data_dir + '/checkpoint-{epoch}.h5'))
        #logdir = args.output_data_dir + '/' + datetime.now().strftime("%Y%m%d-%H%M%S")
        #callbacks.append(TensorBoard(log_dir=logdir))
        #callbacks.append(Sync2S3(logdir=logdir, s3logdir=tensorboard_logs))
        
    #model = get_model(input_shape, lr, weight_decay, optimizer, momentum, hvd)
    # To use ResNet model instead of custom model comment the above line and uncomment the following: 
    #model = get_resnet_model(input_shape, lr, weight_decay, optimizer, momentum, hvd)

    # Train model
    # Change 7: Update the number of steps/epoch
    
    import time
    
    start_time = time.time()
    
    print("A", NUM_TRAIN_IMAGES, GLOBAL_BATCH_SIZE, size)
    print("B", NUM_TRAIN_IMAGES // GLOBAL_BATCH_SIZE)
    print("C", (NUM_TRAIN_IMAGES // GLOBAL_BATCH_SIZE) // size)
    
    history = model.fit(dataset,
                        #steps_per_epoch = (NUM_TRAIN_IMAGES // GLOBAL_BATCH_SIZE) // size,
                        steps_per_epoch = NUM_TRAIN_IMAGES // GLOBAL_BATCH_SIZE,
                        #validation_data = (test_images, test_labels),
                        #validation_steps = (NUM_VALID_IMAGES // batch_size) // size,
                        verbose = 2 if hvd.rank() == 0 else 2,
                        epochs = EPOCHS, 
                        callbacks = callbacks)

    # Evaluate model performance
    if hvd.rank() == 0:
        score = model.evaluate((test_images, test_labels),
                               # steps=NUM_TEST_IMAGES // GLOBAL_BATCH_SIZE,
                               steps= 1,
                               verbose=0)

        print('Test loss    :', score[0])
        print('Test accuracy:', score[1])

        print("RUN TIME", time.time() - start_time)
    
    """
    if hvd.rank() == 0:
        save_history("./hvd_history.p", history)
    """        