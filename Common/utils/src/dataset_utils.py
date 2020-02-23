from torchvision import transforms, datasets
import torch
import numpy as np

###################################
#  ImageNet
###################################
def load_imagenet_data( config, args, gpu ):
    """Training set preprocessing and loader"""
    normalize = transforms.Normalize( [ 0.485, 0.456, 0.406 ],
                                      [ 0.229, 0.224, 0.225 ] )
    
    transform = transforms.Compose( [ transforms.RandomResizedCrop( 224 ),
                                      transforms.RandomHorizontalFlip(),
                                      #transforms.ToTensor(),
                                      #normalize 
                                    ] )

    dataset = datasets.ImageFolder( config.train_path, transform=transform )

    if args.gpu is None:
        train_sampler = torch.utils.data.distributed.DistributedSampler( dataset, rank=gpu )
    else:
        train_sampler = None

    return torch.utils.data.DataLoader( dataset, 
                                        batch_size=args.batch_size, 
                                        shuffle=( train_sampler is None ),
                                        num_workers=args.workers,
                                        pin_memory=True,
                                        sampler=train_sampler,
                                        collate_fn=fast_collate 
                                        )

def load_imagenet_val( config, args ):
    """Validation set preprocessing and loader"""
    normalize = transforms.Normalize( [ 0.485, 0.456, 0.406 ],
                                      [ 0.229, 0.224, 0.225 ] )
    
    transform = transforms.Compose( [ transforms.Resize( 224 ),
                                      transforms.CenterCrop( 224 ),
                                      #transforms.ToTensor(),
                                      #normalize 
                                    ] )
    
    valset = datasets.ImageFolder( config.val_path, transform=transform )

    return torch.utils.data.DataLoader( valset, 
                                        batch_size=args.batch_size, 
                                        shuffle=False, 
                                        num_workers=args.workers, 
                                        pin_memory=True,
                                        collate_fn=fast_collate
                                        )

def fast_collate( batch ):
    imgs = [ img[0] for img in batch ]
    targets = torch.tensor( [ target[1] for target in batch], dtype=torch.int64 )
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 3, h, w ), dtype=torch.uint8 ).contiguous( memory_format=torch.contiguous_format )
    for i, img in enumerate( imgs ):
        nump_array = np.asarray( img, dtype=np.uint8 )
        if( nump_array.ndim < 3 ):
            nump_array = np.expand_dims( nump_array, axis=-1 )
        nump_array = np.rollaxis( nump_array, 2 )
        tensor[i] += torch.from_numpy( nump_array )
    return tensor, targets

class data_prefetcher( object ):
    def __init__( self, loader ):
        self.loader = iter( loader )
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor( [ 0.485 * 255, 0.456 * 255, 0.406 * 255 ] ).cuda().view( 1, 3, 1, 1 )
        self.std = torch.tensor( [ 0.229 * 255, 0.224 * 255, 0.225 * 255 ] ).cuda().view( 1, 3, 1, 1 )
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next( self.loader )
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        with torch.cuda.stream( self.stream ):
            self.next_input = self.next_input.cuda( non_blocking=True ).float()
            self.next_input = self.next_input.sub_( self.mean ).div_( self.std )
            self.next_target = self.next_target.cuda( non_blocking=True )

    def __iter__( self ):
        return self

    def __next__( self ):
        torch.cuda.current_stream().wait_stream( self.stream )
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream( torch.cuda.current_stream() )
        else:
            raise StopIteration
        if target is not None:
            target.record_stream( torch.cuda.current_stream() )
        self.preload()
        return input, target

#######################################
# COCO
#######################################
def load_coco_data( config, args, gpu ):
    """Training set preprocessing and loader"""    
    transform = transforms.Compose( [ transforms.ToTensor() ] )

    dataset = datasets.CocoDetection( root=config.train_path, annFile=config.annFile, transform=transform )

    if args.gpu is None:
        train_sampler = torch.utils.data.distributed.DistributedSampler( dataset, rank=gpu )
    else:
        train_sampler = None

    return torch.utils.data.DataLoader( dataset, 
                                        batch_size=args.batch_size, 
                                        shuffle=( train_sampler is None ),
                                        num_workers=args.workers,
                                        pin_memory=True,
                                        sampler=train_sampler )

def load_coco_val( config, args ):
    """Validation set preprocessing and loader"""
    transform = transforms.Compose( [ transforms.ToTensor() ] )
    
    valset = datasets.CocoDetection( config.val_path, annFile=config.annFile, transform=transform )
    return torch.utils.data.DataLoader( valset, 
                                        batch_size=args.batch_size, 
                                        shuffle=False, 
                                        num_workers=args.workers, 
                                        pin_memory=True )