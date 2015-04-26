import numpy as np
import matplotlib.pyplot as plt
import theano
# By convention, the tensor submodule is loaded as T
import theano.tensor as T




def np_gpu_arr(nparray):
    return np.array(nparray, dtype=theano.config.floatX)


# The theano.tensor submodule has various primitive symbolic variable types.
# Here, we're defining a scalar (0-d) variable.
# The argument gives the variable its name.
foo = T.scalar('foo')
# Now, we can define another variable bar which is just foo squared.
bar = foo**2
# It will also be a theano variable.
print type(bar)
print bar.type
# Using theano's pp (pretty print) function, we see that 
# bar is defined symbolically as the square of foo
print theano.pp(bar)



# We can't compute anything with foo and bar yet.
# We need to define a theano function first.
# The first argument of theano.function defines the inputs to the function.
# Note that bar relies on foo, so foo is an input to this function.
# theano.function will compile code for computing values of bar given values of foo
f = theano.function([foo], bar)

print f(3)

print bar.eval({foo: 3})

print f(6)

MS = T.scalar('MS');
A = T.matrix('A')
x = T.vector('x')
b = T.vector('b')
y = T.dot(A, x) + b
# Note that squaring a matrix is element-wise
z = T.sum(A**2)

test_ex = MS**2
# theano.function can compute multiple things at a time
# You can also set default parameter values
# We'll cover theano.config.floatX later
b_default = np.array([0, 0], dtype=theano.config.floatX)
linear_mix = theano.function([A, x, theano.Param(b, default=b_default)], [y, z])

#, theano.Param(b, default=b_default)
test_f = theano.function([theano.Param(MS, default=6)], test_ex)
print test_f()

print test_ex.eval({MS: 11})
#print test_f(4,test_f);#

# Supplying values for A, x, and b

print linear_mix(np.array([[1, 2, 3],
                           [4, 5, 6]], dtype=theano.config.floatX), #A
                 np.array([1, 2, 3], dtype=theano.config.floatX), #x
                 np.array([4, 5], dtype=theano.config.floatX)) #b




shared_var = theano.shared(np.array([[1, 2], [3, 4]], dtype=theano.config.floatX))
# The type of the shared variable is deduced from its initialization
print shared_var.type()



# We can set the value of a shared variable using set_value
shared_var.set_value(np.array([[3, 4], [2, 1]], dtype=theano.config.floatX))
# ..and get it using get_value
print shared_var.get_value()


shared_squared = shared_var**2
# The first argument of theano.function (inputs) tells Theano what the arguments to the compiled function should be.
# Note that because shared_var is shared, it already has a value, so it doesn't need to be an input to the function.
# Therefore, Theano implicitly considers shared_var an input to a function using shared_squared and so we don't need
# to include it in the inputs argument of theano.function.
function_1 = theano.function([], shared_squared)
print function_1()


# We can also update the state of a shared var in a function
subtract = T.matrix('subtract')
# updates takes a dict where keys are shared variables and values are the new value the shared variable should take
# Here, updates will set shared_var = shared_var - subtract
function_2 = theano.function([subtract], shared_var, updates={shared_var: shared_var - subtract})
print "shared_var before subtracting [[1, 1], [1, 1]] using function_2:"
print shared_var.get_value()
# Subtract [[1, 1], [1, 1]] from shared_var


function_2(np.array([[1, 1], [1, 1]], dtype=theano.config.floatX )  )
print "shared_var after calling function_2:"
print shared_var.get_value()
# Note that this also changes the output of function_1, because shared_var is shared!
print "New output of function_1() (shared_var**2):"
print function_1()


# Recall that bar = foo**2
# We can compute the gradient of bar with respect to foo like so:
bar_grad = T.grad(bar, foo)
# We expect that bar_grad = 2*foo
print(bar_grad.eval({foo: 10}))


# Recall that y = Ax + b
# We can also compute a Jacobian like so:
y_J = theano.gradient.jacobian(y, x)
linear_mix_J = theano.function([A, x, b], y_J)
# Because it's a linear mix, we expect the output to always be A
print linear_mix_J(np_gpu_arr([[9, 8, 7], [4, 5, 6]]), #A
                   np_gpu_arr([1, 2, 3]), #x
                   np_gpu_arr([4, 5])) #b
# We can also compute the Hessian with theano.gradient.hessian (skipping that here)



# A simple division function
num = T.scalar('num')
den = T.scalar('den')
divide = theano.function([num, den], num/den)
print divide(10, 2)
# This will cause a NaN
print divide(0, 0)



# To compile a function in debug mode, just set mode='DebugMode'
divide = theano.function([num, den], num/den, mode='DebugMode')
# NaNs now cause errors
print divide(0, 0)


print theano.config.device
print theano.config.floatX