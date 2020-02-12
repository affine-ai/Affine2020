from torchvision import transforms, datasets
import torch


###################################
#  ImageNet
###################################
def load_imagenet_data( config, args, gpu ):
    """Training set preprocessing and loader"""
    normalize = transforms.Normalize( [ 0.485, 0.456, 0.406 ],
                                      [ 0.229, 0.224, 0.225 ] )
    
    transform = transforms.Compose( [ transforms.RandomResizedCrop( 224 ), \
                                      transforms.RandomHorizontalFlip(), \
                                      transforms.ToTensor(),
                                      normalize ] )

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
                                        sampler=train_sampler )

def load_imagenet_val( config, args ):
    """Validation set preprocessing and loader"""
    normalize = transforms.Normalize( [ 0.485, 0.456, 0.406 ],
                                      [ 0.229, 0.224, 0.225 ] )
    
    transform = transforms.Compose( [ transforms.Resize( 224 ),
                                      transforms.CenterCrop( 224 ),
                                      transforms.ToTensor(),
                                      normalize ] )
    
    valset = datasets.ImageFolder( config.val_path, transform=transform )
    return torch.utils.data.DataLoader( valset, 
                                        batch_size=args.batch_size, 
                                        shuffle=False, 
                                        num_workers=args.workers, 
                                        pin_memory=True )

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