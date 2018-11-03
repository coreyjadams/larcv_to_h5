import sys, os

from ROOT import larcv
larcv.load_pyutil()
import h5py
import numpy


def main(input_file_name):

    # path     = os.path.basename(input_file_name)
    basename = os.path.basename(input_file_name)

    print basename

    # Open this file with larcv and see what is inside:
    larcv_io = larcv.IOManager()
    larcv_io.add_in_file(input_file_name)
    larcv_io.initialize()
    # larcv_io.read_entry(0)

    output_name = basename.rstrip('.root') + '.h5'
    print output_name
    if os.path.exists(output_name):
        os.remove(output_name)


    _h5_out = h5py.File(output_name, 'a')


    products = larcv_io.product_list()

    n_entries = larcv_io.get_n_entries()

    for i in range(n_entries):

        print("Processing entry {}".format(i))

        larcv_io.read_entry(i)

        for product in products:
            producers = larcv_io.producer_list(product)
            for producer in producers:
                if product == 'image2d':
                    convert_image2d(larcv_io, _h5_out, producer)
                if product == 'particle':
                    convert_particle(larcv_io, _h5_out, producer)
        convert_eventid(larcv_io, _h5_out)


        _h5_out.flush()

    _h5_out.close()

def convert_eventid(larcv_io, _h5_out):

    # Store the event id and entry information:
    event_id_dtype = numpy.dtype(
        [   ('entry',   numpy.uint32),
            ('event',   numpy.uint32),
            ('subrun',  numpy.uint32),
            ('run',     numpy.uint32),
        ]
    )

    if 'eventid' not in _h5_out:
        eventid = _h5_out.create_dataset('eventid', dtype=event_id_dtype,
                                         shape=(0,), maxshape=(1000000,),
                                         compression='gzip', chunks=True)

    else:
        eventid = _h5_out['eventid']


    # Add a row to this eventid table:
    eventid.resize((eventid.shape[0] + 1,))
    eventid[-1,'entry'] = larcv_io.current_entry()
    eventid[-1,'event'] = larcv_io.event_id().event()
    eventid[-1,'run']   = larcv_io.event_id().run()
    eventid[-1,'subrun'] = larcv_io.event_id().subrun()

    return

# Create a function to convert image2d
def convert_image2d(larcv_io, _h5_out, producer):

    # Image2d lives a a group, which also contains a table for image2d metadata
    # In particular, the metadata table contains a list of producers as well as the
    # number of images for each producer

    # Get the data pointer in larcv:
    image2d = larcv_io.get_data("image2d", producer)


    if 'image2d' in _h5_out:
        group = _h5_out['image2d']
    else:
        group = _h5_out.create_group('image2d')


    # Define the dtype of the metadata to be stored:
    metadata_dtype = numpy.dtype([('producer_name', 'S25'),
                                  ('nplanes', numpy.uint8),
                                  ('nentries', numpy.uint32)])

    # Get the metadata table:
    if 'metadata' in group:
        metadata = group['metadata']
    else:
        metadata = group.create_dataset('metadata', dtype=metadata_dtype,
                                        shape=(0,), maxshape=(1000000,),
                                        compression='gzip', chunks=True)

    # If this producer is not in the metadata table, add it:
    # Get the list of producers:
    stored_producers = metadata['producer_name']
    if producer not in stored_producers:
        # Make room for the metadata:
        curr_size = len(metadata)
        metadata.resize((curr_size+ 1,))
        # Create the metadata entry:
        this_metadata = {}
        metadata[-1,'producer_name'] = producer
        metadata[-1,'nplanes'] = image2d.as_vector().size()
        metadata[-1,'nentries'] = 0




    if producer in group:
        subgroup = group[producer]
    else:
        subgroup = group.create_group(producer)



    for image in image2d.as_vector():

        dataset_name =  "image_{}".format(image.meta().id())

        # if this dataset does not exist, create it:
        if dataset_name not in subgroup:
            shape     = (1,image.meta().rows(), image.meta().cols() )
            max_shape = (None,image.meta().rows(), image.meta().cols() )
            dset = subgroup.create_dataset(dataset_name, shape=shape,
                                           maxshape=max_shape,
                                           compression='gzip', chunks=True)

            # Add the attributes for this dataset:
            origin = numpy.asarray((image.meta().origin().x, image.meta().origin().y ))
            top_right = numpy.asarray((image.meta().top_right().x, image.meta().top_right().y ))

            dset.attrs['id']        = image.meta().id()
            dset.attrs['origin']    = origin
            dset.attrs['top_right'] = top_right
            dset.attrs['rows']      = image.meta().rows()
            dset.attrs['cols']      = image.meta().cols()
            dset.attrs['unit']      = image.meta().unit()

        else:
            dset = subgroup[dataset_name]

        # Make sure the entry_id (larcv) write to the proper index in h5:
        entry = larcv_io.current_entry()

        # if the dataset is not big enough, resize:
        if len(dset) <= entry:
            curr_shape = list(dset.shape)
            curr_shape[0] += 1
            dset.resize(curr_shape)

        # Get the image 2d array object:
        image_data = larcv.as_ndarray(image)

        # Write the output data into the dataset:
        dset[-1,:] = image_data

    # Update the number of entries in the metadata table
    producers = metadata['producer_name']
    i = 0
    for p in producers:
        if p == producer:
            metadata[i,'nentries'] += 1
            break
        i += 1

    return
    # _larcv_out = basename + "_larcv.root"

def convert_particle(larcv_io, _h5_out, producer):

    # particle information gets stored as a numpy structured array
    # So, the dtype is fixed for each particle and we can define it directly:
    particle_dtype = numpy.dtype(
        [   ('id',                numpy.uint16),
            ('mcst_index',        numpy.uint16),
            ('mct_index',         numpy.uint16),
            ('current_type',      numpy.uint16),
            ('interaction_type',  numpy.uint16),
            ('track_id',          numpy.uint16),
            ('pdg',               numpy.int16),
            ('px',                numpy.float64),
            ('py',                numpy.float64),
            ('pz',                numpy.float64),
            ('vx',                numpy.float64),
            ('vy',                numpy.float64),
            ('vz',                numpy.float64),
            ('vt',                numpy.float64),
            ('parent_track_id',   numpy.uint16),
            ('parent_pdg',        numpy.uint16),
            ('ancestor_track_id', numpy.uint16),
            ('ancestor_pdg',      numpy.uint16),
        ]
    )

    extents_dtype = numpy.dtype(
        [   ('entry',       numpy.uint32),
            ('first_idx',   numpy.uint32),
            ('last_idx',    numpy.uint32),
        ]
    )

    particle_set = larcv_io.get_data('particle', producer)



    if 'particle' in _h5_out:
        group = _h5_out['particle']
    else:
        group = _h5_out.create_group('particle')


    # Define the dtype of the metadata to be stored:
    metadata_dtype = numpy.dtype([('producer_name', 'S25'),
                                  ('nentries', numpy.uint32)])

    # Get the metadata table:
    if 'metadata' in group:
        metadata = group['metadata']
    else:
        metadata = group.create_dataset('metadata', dtype=metadata_dtype,
                                        shape=(0,), maxshape=(1000000,),
                                        compression='gzip', chunks=True)

    # If this producer is not in the metadata table, add it:
    # in the metadata table, add it:
    # Get the list of producers:
    stored_producers = metadata['producer_name']
    if producer not in stored_producers:
        # Make room for the metadata:
        curr_size = len(metadata)
        metadata.resize((curr_size+ 1,))
        # Create the metadata entry:
        this_metadata = {}
        metadata[-1,'producer_name'] = producer
        metadata[-1,'nentries'] = 0

    # For each producer, create a subgroup containing the list
    # of particle objects as well as an extents table,
    # which defines which particles go to which event

    if producer in group:
        subgroup = group[producer]
    else:
        subgroup = group.create_group(producer)

    # Next, get the dset for this set of particles:

    # if this dataset does not exist, create it:
    if 'particles' not in subgroup:
        dset = subgroup.create_dataset('particles', dtype=particle_dtype,
                                       shape = (0,), maxshape=(1000000,),
                                       compression='gzip', chunks=True)
    else:
        dset = subgroup['particles']

    # And, get the extents table or create it:
    if 'extents' not in subgroup:
        extents = subgroup.create_dataset('extents', dtype=extents_dtype,
                                          shape = (0,), maxshape=(1000000,),
                                          compression='gzip',chunks=True)
    else:
        extents = subgroup['extents']

    # Create an array of all of the particles:
    n_particles = particle_set.as_vector().size()

    # pad the output dataset appropriately:
    offset = dset.shape[0]
    dset.resize((offset + n_particles,))


    i = 0
    for particle in particle_set.as_vector():
        # Create the dataset, if necessary:
        dset[offset + i,'id'] = particle.id()
        dset[offset + i,'mcst_index'] = particle.mcst_index()
        dset[offset + i,'mct_index'] = particle.mct_index()
        dset[offset + i,'current_type'] = particle.nu_current_type()
        dset[offset + i,'interaction_type'] = particle.nu_interaction_type()
        dset[offset + i,'track_id'] = particle.track_id()
        dset[offset + i,'pdg'] = particle.pdg_code()
        dset[offset + i,'px'] = particle.px()
        dset[offset + i,'py'] = particle.py()
        dset[offset + i,'pz'] = particle.pz()
        dset[offset + i,'vx'] = particle.x()
        dset[offset + i,'vy'] = particle.y()
        dset[offset + i,'vz'] = particle.z()
        dset[offset + i,'vt'] = particle.t()
        dset[offset + i,'parent_track_id'] = particle.parent_track_id()
        dset[offset + i,'parent_pdg'] = particle.parent_pdg_code()
        dset[offset + i,'ancestor_track_id'] = particle.ancestor_track_id()
        dset[offset + i,'ancestor_pdg'] = particle.ancestor_pdg_code()

        i += 1

    # Update the extents table:
    extents.resize((extents.shape[0] + 1,))
    extents[-1, 'entry']     = larcv_io.current_entry()
    extents[-1, 'first_idx'] = offset
    extents[-1, 'last_idx']  = offset + n_particles - 1


    # Lastly, update the particle metadata table to include another entry in this
    # producer table:
    producers = metadata['producer_name']
    i = 0
    for p in producers:
        if p == producer:
            metadata[i,'nentries'] += 1
            break
        i += 1

if __name__ == '__main__':
    main(input_file_name = "/home/cadams/DeepLearnPhysics/larcv_open_data/practice_train_5k_rand.root")

