import os
import random
import scipy.io
import shutil


def parse_mat( rootpath ):
    mat_file = os.path.join( rootpath, "data", "meta_clsloc.mat" )
    mat = scipy.io.loadmat( mat_file, squeeze_me=True )[ 'synsets']
    nums_children = list( zip( *mat ) )[ 4 ]
    mat = [ mat[ i ] for i, num_children in enumerate( nums_children )
            if num_children == 0]
    idcs, wnids, classes = list( zip( *mat ) )[:3]
    classes = [ tuple( clss.split( ', ' ) ) for clss in classes ]
    idx_to_wnid = { idx: wnid for idx, wnid in zip( idcs, wnids ) }
    wnid_to_classes = { wnid: clss for wnid, clss in zip( wnids, classes ) }
    return idx_to_wnid, wnid_to_classes


def process_val_set( rootpath ):
    idx_to_wnid, wnid_to_classes = parse_mat( rootpath )
    val_idcs = parse_validation_ground_truth( rootpath )
    val_wnids = [ idx_to_wnid[ idx ] for idx in val_idcs ]
    return wnid_to_classes, val_wnids


def parse_validation_ground_truth( rootpath, path='data',
                          filename='ILSVRC2014_clsloc_validation_ground_truth.txt'):
    with open(os.path.join( rootpath, path, filename ), 'r') as f:
        val_idcs = f.readlines()
    return [ int( val_idx ) for val_idx in val_idcs ]


def prepare_val_folder( folder, wnids ):
    img_files = sorted( [ os.path.join( folder, file ) \
                                for file in os.listdir( folder ) ] )

    for wnid in set( wnids ):
        os.mkdir( os.path.join( folder, wnid ) )

    for wnid, img_file in zip( wnids, img_files ):
        shutil.move( img_file, os.path.join( folder, wnid, os.path.basename( img_file ) ) )

_, wnids = process_val_set( "/home/vipul/Downloads/ImageNet/" )
prepare_val_folder( "/home/vipul/Downloads/ImageNet/val/", wnids )
