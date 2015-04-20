import time
import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from logistic_sgd import load_data
from theano import function, config, shared, sandbox
import scipy.io
#from matplotlib.matlab import *

from utils import tile_raster_images
import matplotlib.pyplot as plt

try:
    import PIL.Image as Image
except ImportError:
    import Image


if __name__ == '__main__':
    mat = scipy.io.loadmat('all_target1.mat')
    import dA
    learning_rate=0.1
    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    batch_size=20
    training_epochs=30
    
    
    x = T.matrix('x')  # the data is presented as rasterized images
    index = T.lscalar()    # index to a [mini]batch

    #da = dA.dA(
    #    numpy_rng=rng,
    #    theano_rng=theano_rng,
    #    input=x,
    #    n_visible=28 * 28,
    #    n_hidden=500
    #)

    da = dA.dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=4400,
        n_hidden=250
    )


    cost, updates = da.get_cost_updates(
        corruption_level=0.7,
        learning_rate=learning_rate
    )


    #datasets = load_data('mnist.pkl.gz')
    #train_set_x, train_set_y = datasets[0]

    train_set_x2  = shared(numpy.asarray(mat['all_target_faltten1'][:,0:4400],config.floatX))

    n_train_batches = train_set_x2.get_value(borrow=True).shape[0] / batch_size


    #train_set_x2  = shared(numpy.asarray(train_set_x.get_value(borrow=True)[:,0:28], config.floatX))
    
    
    
    
    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x2[index * batch_size: (index + 1) * batch_size]
        }
    )

    start_time = time.clock()

     # go through training epochs
    for epoch in xrange(training_epochs):
        # go through trainng set
        c = []
        print(time.clock() - start_time)
        for batch_index in xrange(n_train_batches):            
            c.append(train_da(batch_index))
            
        print 'Training epoch %d, cost ' % epoch, numpy.mean(c)
        

    #image = Image.fromarray(tile_raster_images(X=da.W.get_value(borrow=True).T,img_shape=(1, 28), tile_shape=(5, 5),tile_spacing=(3, 3)))
    ##dA.test_dA()
    #image.save('stam.png')
    #image.show();

    for i in range(10):
        plt.subplot(2,5,i+1) 
        plt.plot(da.W.get_value()[:,i].T)
    
    plt.show()

    

    # difference of Gaussians
    im = plt.imshow(da.W.get_value().T)
    im.set_interpolation('nearest')
    plt.axis('off')
    plt.show()
    #
    print ('hi world2')
    

