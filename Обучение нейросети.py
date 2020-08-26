from os import listdir
from xml.etree import ElementTree

from certifi.__main__ import args
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.model import MaskRCNN



class TtsDataset(Dataset):
    # load the dataset definitions
    def load_dataset(self, dataset_dir, is_train=True):
        # define one class
        self.add_class("dataset", 1, "bus")
        self.add_class("dataset", 2, "truck")
        self.add_class("dataset", 3, "car")
        self.add_class("dataset", 4, "bmw")
        self.add_class("dataset", 5, "toyota")
        self.add_class("dataset", 6, "peugeot")
        self.add_class("dataset", 7, "kia")
        self.add_class("dataset", 8, "zonda")
        self.add_class("dataset", 9, "man")
        self.add_class("dataset", 10, "hyundai")
        self.add_class("dataset", 11, "scania")

        # define data locations
        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annots/'
        # find all images
        for filename in listdir(images_dir):
            # extract image id
            image_id = filename[:-4]
            # skip bad images
            if image_id in ['00090']:
                continue
            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.xml'
            # add to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path,
                           class_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    # extract bounding boxes from an annotation file
    def extract_boxes(self, filename):
        # load and parse the file
        tree = ElementTree.parse(filename)
        # get the root of the document
        root = tree.getroot()
        # extract each bounding box
        boxes = list()
        for box in root.findall('.//object'):
            name = box.find('name').text
            xmin = int(box.find('./bndbox/xmin').text)
            ymin = int(box.find('./bndbox/ymin').text)
            xmax = int(box.find('./bndbox/xmax').text)
            ymax = int(box.find('./bndbox/ymax').text)
            coors = [xmin, ymin, xmax, ymax, name]
            boxes.append(coors)
        # extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

    # load the masks for an image
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        # define box file location
        path = info['annotation']
        # load XML
        boxes, w, h = self.extract_boxes(path)
        # create one array for all masks, each on a different channel
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        # create masks
        class_ids = list()

        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            if (box[4] == 'bus'):
                masks[row_s:row_e, col_s:col_e, i] = 1                
                class_ids.append(self.class_names.index('bus'))
            else:
                if (box[4] == 'truck'):
                    masks[row_s:row_e, col_s:col_e, i] = 2                
                    class_ids.append(self.class_names.index('truck'))
                else:
                    if (box[4] == 'car'):
                        masks[row_s:row_e, col_s:col_e, i] = 3                
                        class_ids.append(self.class_names.index('car'))
                    else:
                        if (box[4] == 'bmw'):
                            masks[row_s:row_e, col_s:col_e, i] = 4                
                            class_ids.append(self.class_names.index('bmw'))
                        else:
                            if (box[4] == 'toyota'):
                                masks[row_s:row_e, col_s:col_e, i] = 5
                                class_ids.append(self.class_names.index('toyota'))
                            else:
                                if (box[4] == 'peugeot'):
                                    masks[row_s:row_e, col_s:col_e, i] = 6
                                    class_ids.append(self.class_names.index('peugeot'))
                                else:
                                    if (box[4] == 'kia'):
                                        masks[row_s:row_e, col_s:col_e, i] = 7
                                        class_ids.append(self.class_names.index('kia'))
                                    else:
                                        if (box[4] == 'zonda'):
                                            masks[row_s:row_e, col_s:col_e, i] = 8
                                            class_ids.append(self.class_names.index('zonda'))
                                        else:
                                            if (box[4] == 'man'):
                                                masks[row_s:row_e, col_s:col_e, i] = 9
                                                class_ids.append(self.class_names.index('man'))
                                            else:
                                                if (box[4] == 'hyundai'):
                                                    masks[row_s:row_e, col_s:col_e, i] = 10
                                                    class_ids.append(self.class_names.index('hyundai'))
                                                else:
                                                    if (box[4] == 'scania'):
                                                        masks[row_s:row_e, col_s:col_e, i] = 11
                                                        class_ids.append(self.class_names.index('scania'))
        return masks, asarray(class_ids, dtype='int32')

        # load an image reference
        def image_reference(self, image_id):
            info = self.image_info[image_id]
            return info['path']


# define a configuration for the model
class TtsConfig(Config):
    # define the name of the configuration
    NAME = "tts_cfg"
    # number of classes (background + all)
    NUM_CLASSES = 1 + 11
    # number of training steps per epoch
    STEPS_PER_EPOCH = 131


# prepare train set
train_set = TtsDataset()
train_set.load_dataset('bus', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))

# prepare train set
train_set = TtsDataset()
train_set.load_dataset('truck', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))

# prepare train set
train_set = TtsDataset()
train_set.load_dataset('car', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))

# prepare train set
train_set = TtsDataset()
train_set.load_dataset('bmw', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))

# prepare train set
train_set = TtsDataset()
train_set.load_dataset('bus', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))

# prepare train set
train_set = TtsDataset()
train_set.load_dataset('toyota', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))

# prepare train set
train_set = TtsDataset()
train_set.load_dataset('peugeot', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))

# prepare train set
train_set = TtsDataset()
train_set.load_dataset('kia', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))

# prepare train set
train_set = TtsDataset()
train_set.load_dataset('zonda', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))

# prepare train set
train_set = TtsDataset()
train_set.load_dataset('man', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))

# prepare train set
train_set = TtsDataset()
train_set.load_dataset('hyundai', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))

# prepare train set
train_set = TtsDataset()
train_set.load_dataset('scania', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))


# prepare config
config = TtsConfig()
config.display()
# define the model
model = MaskRCNN(mode='training', model_dir='./', config=config)
# load weights (mscoco) and exclude the output layers
#if args.weights.lower() == "coco":
model.load_weights('mask_rcnn_coco.h5', by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
#else:
#    model.load_weights(weights_path, by_name=True)
# train weights (output layers or 'heads')
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')

