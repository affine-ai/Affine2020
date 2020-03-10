from torchvision import transforms, datasets
import torch
import numpy as np
import matplotlib.pyplot as plt


class data_prefetcher( object ):
    def __init__( self, loader ):
        self.loader = iter( loader )
        self.stream = torch.cuda.Stream()
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


###################################
#  ImageNet
###################################
#normalize = transforms.Normalize( [ 0.45, 0.45, 0.45 ],
#                                    [ 0.225, 0.225, 0.225 ] )
normalize = transforms.Normalize( [ 0.485, 0.456, 0.406 ],
                                 [ 0.229, 0.224, 0.225 ] )
def load_imagenet_data( path, args, hyper, distributed ):
    """Training set preprocessing and loader
    """
    transform = transforms.Compose( [ transforms.RandomResizedCrop( 224 ),
                                      transforms.RandomHorizontalFlip(),
                                      #transforms.Grayscale( num_output_channels=3 ),
                                      transforms.ToTensor(),
                                      normalize 
                                    ] )

    dataset = datasets.ImageFolder( path, transform=transform )

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler( dataset )
    else:
        train_sampler = None

    return torch.utils.data.DataLoader( dataset, 
                                        batch_size=hyper.batch_size, 
                                        shuffle=( train_sampler is None ),
                                        num_workers=args.workers,
                                        pin_memory=True,
                                        sampler=train_sampler
                                        )

def load_imagenet_val( path, args, hyper, distributed ):
    """Validation set preprocessing and loader
    """
    transform = transforms.Compose( [ transforms.Resize( 224 ),
                                      transforms.CenterCrop( 224 ),
                                      transforms.ToTensor(),
                                      normalize
                                    ] )
    
    valset = datasets.ImageFolder( path, transform=transform )

    return torch.utils.data.DataLoader( valset, 
                                        batch_size=hyper.batch_size, 
                                        shuffle=False, 
                                        num_workers=args.workers, 
                                        pin_memory=True
                                        )

#######################################
# COCO
#######################################
def load_coco_data( config, args, hyper, distributed ):
    """Training set preprocessing and loader
    """    
    transform = transforms.Compose( [ transforms.ToTensor() ] )

    dataset = datasets.CocoDetection( root=config.train_path, annFile=config.annFile, transform=transform )

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler( dataset )
    else:
        train_sampler = None

    return torch.utils.data.DataLoader( dataset, 
                                        batch_size=hyper.batch_size, 
                                        shuffle=( train_sampler is None ),
                                        num_workers=args.workers,
                                        pin_memory=True,
                                        sampler=train_sampler )

def load_coco_val( config, args, hyper, distributed ):
    """Validation set preprocessing and loader
    """
    transform = transforms.Compose( [ transforms.ToTensor() ] )
    
    valset = datasets.CocoDetection( config.val_path, annFile=config.annFile, transform=transform )
    return torch.utils.data.DataLoader( valset, 
                                        batch_size=hyper.batch_size, 
                                        shuffle=False, 
                                        num_workers=args.workers, 
                                        pin_memory=True )