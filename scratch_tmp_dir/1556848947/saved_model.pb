í
ůŰ
:
Add
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
ě
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape
9
VarIsInitializedOp
resource
is_initialized
"serve*1.12.02
b'unknown'Üş

conv2d_inputPlaceholder*$
shape:˙˙˙˙˙˙˙˙˙@@*
dtype0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@@
Š
.conv2d/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:* 
_class
loc:@conv2d/kernel*%
valueB"         @   

,conv2d/kernel/Initializer/random_uniform/minConst* 
_class
loc:@conv2d/kernel*
valueB
 *Ş7˝*
dtype0*
_output_shapes
: 

,conv2d/kernel/Initializer/random_uniform/maxConst* 
_class
loc:@conv2d/kernel*
valueB
 *Ş7=*
dtype0*
_output_shapes
: 
đ
6conv2d/kernel/Initializer/random_uniform/RandomUniformRandomUniform.conv2d/kernel/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
:@*

seed *
T0* 
_class
loc:@conv2d/kernel*
seed2 
Ň
,conv2d/kernel/Initializer/random_uniform/subSub,conv2d/kernel/Initializer/random_uniform/max,conv2d/kernel/Initializer/random_uniform/min* 
_class
loc:@conv2d/kernel*
_output_shapes
: *
T0
ě
,conv2d/kernel/Initializer/random_uniform/mulMul6conv2d/kernel/Initializer/random_uniform/RandomUniform,conv2d/kernel/Initializer/random_uniform/sub*
T0* 
_class
loc:@conv2d/kernel*&
_output_shapes
:@
Ţ
(conv2d/kernel/Initializer/random_uniformAdd,conv2d/kernel/Initializer/random_uniform/mul,conv2d/kernel/Initializer/random_uniform/min*
T0* 
_class
loc:@conv2d/kernel*&
_output_shapes
:@
ą
conv2d/kernelVarHandleOp*
	container *
shape:@*
dtype0*
_output_shapes
: *
shared_nameconv2d/kernel* 
_class
loc:@conv2d/kernel
k
.conv2d/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d/kernel*
_output_shapes
: 

conv2d/kernel/AssignAssignVariableOpconv2d/kernel(conv2d/kernel/Initializer/random_uniform* 
_class
loc:@conv2d/kernel*
dtype0

!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel* 
_class
loc:@conv2d/kernel*
dtype0*&
_output_shapes
:@

conv2d/bias/Initializer/zerosConst*
_class
loc:@conv2d/bias*
valueB@*    *
dtype0*
_output_shapes
:@

conv2d/biasVarHandleOp*
shared_nameconv2d/bias*
_class
loc:@conv2d/bias*
	container *
shape:@*
dtype0*
_output_shapes
: 
g
,conv2d/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d/bias*
_output_shapes
: 

conv2d/bias/AssignAssignVariableOpconv2d/biasconv2d/bias/Initializer/zeros*
_class
loc:@conv2d/bias*
dtype0

conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_class
loc:@conv2d/bias*
dtype0*
_output_shapes
:@
e
conv2d/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
r
conv2d/Conv2D/ReadVariableOpReadVariableOpconv2d/kernel*
dtype0*&
_output_shapes
:@
ě
conv2d/Conv2DConv2Dconv2d_inputconv2d/Conv2D/ReadVariableOp*/
_output_shapes
:˙˙˙˙˙˙˙˙˙==@*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
e
conv2d/BiasAdd/ReadVariableOpReadVariableOpconv2d/bias*
dtype0*
_output_shapes
:@

conv2d/BiasAddBiasAddconv2d/Conv2Dconv2d/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*/
_output_shapes
:˙˙˙˙˙˙˙˙˙==@
]
conv2d/ReluReluconv2d/BiasAdd*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙==@
­
0conv2d_1/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@conv2d_1/kernel*%
valueB"      @   @   *
dtype0*
_output_shapes
:

.conv2d_1/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel*
valueB
 *×ł]˝*
dtype0

.conv2d_1/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@conv2d_1/kernel*
valueB
 *×ł]=*
dtype0*
_output_shapes
: 
ö
8conv2d_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_1/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0*&
_output_shapes
:@@*

seed *
T0*"
_class
loc:@conv2d_1/kernel
Ú
.conv2d_1/kernel/Initializer/random_uniform/subSub.conv2d_1/kernel/Initializer/random_uniform/max.conv2d_1/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: 
ô
.conv2d_1/kernel/Initializer/random_uniform/mulMul8conv2d_1/kernel/Initializer/random_uniform/RandomUniform.conv2d_1/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@@
ć
*conv2d_1/kernel/Initializer/random_uniformAdd.conv2d_1/kernel/Initializer/random_uniform/mul.conv2d_1/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@@
ˇ
conv2d_1/kernelVarHandleOp*
_output_shapes
: * 
shared_nameconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*
	container *
shape:@@*
dtype0
o
0conv2d_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_1/kernel*
_output_shapes
: 

conv2d_1/kernel/AssignAssignVariableOpconv2d_1/kernel*conv2d_1/kernel/Initializer/random_uniform*"
_class
loc:@conv2d_1/kernel*
dtype0

#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*
dtype0*&
_output_shapes
:@@*"
_class
loc:@conv2d_1/kernel

conv2d_1/bias/Initializer/zerosConst* 
_class
loc:@conv2d_1/bias*
valueB@*    *
dtype0*
_output_shapes
:@
Ľ
conv2d_1/biasVarHandleOp*
shape:@*
dtype0*
_output_shapes
: *
shared_nameconv2d_1/bias* 
_class
loc:@conv2d_1/bias*
	container 
k
.conv2d_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_1/bias*
_output_shapes
: 

conv2d_1/bias/AssignAssignVariableOpconv2d_1/biasconv2d_1/bias/Initializer/zeros*
dtype0* 
_class
loc:@conv2d_1/bias

!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias* 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
:@
g
conv2d_1/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
v
conv2d_1/Conv2D/ReadVariableOpReadVariableOpconv2d_1/kernel*
dtype0*&
_output_shapes
:@@
ď
conv2d_1/Conv2DConv2Dconv2d/Reluconv2d_1/Conv2D/ReadVariableOp*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
	dilations
*
T0
i
conv2d_1/BiasAdd/ReadVariableOpReadVariableOpconv2d_1/bias*
dtype0*
_output_shapes
:@

conv2d_1/BiasAddBiasAddconv2d_1/Conv2Dconv2d_1/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
a
conv2d_1/ReluReluconv2d_1/BiasAdd*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
T0
e
dropout/IdentityIdentityconv2d_1/Relu*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
T0
­
0conv2d_2/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@conv2d_2/kernel*%
valueB"      @      *
dtype0*
_output_shapes
:

.conv2d_2/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *"
_class
loc:@conv2d_2/kernel*
valueB
 *ó5˝

.conv2d_2/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@conv2d_2/kernel*
valueB
 *ó5=*
dtype0*
_output_shapes
: 
÷
8conv2d_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_2/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0*'
_output_shapes
:@*

seed *
T0*"
_class
loc:@conv2d_2/kernel
Ú
.conv2d_2/kernel/Initializer/random_uniform/subSub.conv2d_2/kernel/Initializer/random_uniform/max.conv2d_2/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_2/kernel*
_output_shapes
: 
ő
.conv2d_2/kernel/Initializer/random_uniform/mulMul8conv2d_2/kernel/Initializer/random_uniform/RandomUniform.conv2d_2/kernel/Initializer/random_uniform/sub*'
_output_shapes
:@*
T0*"
_class
loc:@conv2d_2/kernel
ç
*conv2d_2/kernel/Initializer/random_uniformAdd.conv2d_2/kernel/Initializer/random_uniform/mul.conv2d_2/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_2/kernel*'
_output_shapes
:@
¸
conv2d_2/kernelVarHandleOp* 
shared_nameconv2d_2/kernel*"
_class
loc:@conv2d_2/kernel*
	container *
shape:@*
dtype0*
_output_shapes
: 
o
0conv2d_2/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_2/kernel*
_output_shapes
: 

conv2d_2/kernel/AssignAssignVariableOpconv2d_2/kernel*conv2d_2/kernel/Initializer/random_uniform*"
_class
loc:@conv2d_2/kernel*
dtype0
 
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*'
_output_shapes
:@*"
_class
loc:@conv2d_2/kernel*
dtype0

conv2d_2/bias/Initializer/zerosConst*
_output_shapes	
:* 
_class
loc:@conv2d_2/bias*
valueB*    *
dtype0
Ś
conv2d_2/biasVarHandleOp* 
_class
loc:@conv2d_2/bias*
	container *
shape:*
dtype0*
_output_shapes
: *
shared_nameconv2d_2/bias
k
.conv2d_2/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_2/bias*
_output_shapes
: 

conv2d_2/bias/AssignAssignVariableOpconv2d_2/biasconv2d_2/bias/Initializer/zeros*
dtype0* 
_class
loc:@conv2d_2/bias

!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes	
:* 
_class
loc:@conv2d_2/bias*
dtype0
g
conv2d_2/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
w
conv2d_2/Conv2D/ReadVariableOpReadVariableOpconv2d_2/kernel*
dtype0*'
_output_shapes
:@
ő
conv2d_2/Conv2DConv2Ddropout/Identityconv2d_2/Conv2D/ReadVariableOp*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
j
conv2d_2/BiasAdd/ReadVariableOpReadVariableOpconv2d_2/bias*
dtype0*
_output_shapes	
:

conv2d_2/BiasAddBiasAddconv2d_2/Conv2Dconv2d_2/BiasAdd/ReadVariableOp*
data_formatNHWC*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
b
conv2d_2/ReluReluconv2d_2/BiasAdd*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
­
0conv2d_3/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@conv2d_3/kernel*%
valueB"            *
dtype0*
_output_shapes
:

.conv2d_3/kernel/Initializer/random_uniform/minConst*"
_class
loc:@conv2d_3/kernel*
valueB
 *qÄ˝*
dtype0*
_output_shapes
: 

.conv2d_3/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@conv2d_3/kernel*
valueB
 *qÄ=*
dtype0*
_output_shapes
: 
ř
8conv2d_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_3/kernel/Initializer/random_uniform/shape*
dtype0*(
_output_shapes
:*

seed *
T0*"
_class
loc:@conv2d_3/kernel*
seed2 
Ú
.conv2d_3/kernel/Initializer/random_uniform/subSub.conv2d_3/kernel/Initializer/random_uniform/max.conv2d_3/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*"
_class
loc:@conv2d_3/kernel
ö
.conv2d_3/kernel/Initializer/random_uniform/mulMul8conv2d_3/kernel/Initializer/random_uniform/RandomUniform.conv2d_3/kernel/Initializer/random_uniform/sub*(
_output_shapes
:*
T0*"
_class
loc:@conv2d_3/kernel
č
*conv2d_3/kernel/Initializer/random_uniformAdd.conv2d_3/kernel/Initializer/random_uniform/mul.conv2d_3/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_3/kernel*(
_output_shapes
:
š
conv2d_3/kernelVarHandleOp*
	container *
shape:*
dtype0*
_output_shapes
: * 
shared_nameconv2d_3/kernel*"
_class
loc:@conv2d_3/kernel
o
0conv2d_3/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_3/kernel*
_output_shapes
: 

conv2d_3/kernel/AssignAssignVariableOpconv2d_3/kernel*conv2d_3/kernel/Initializer/random_uniform*"
_class
loc:@conv2d_3/kernel*
dtype0
Ą
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*"
_class
loc:@conv2d_3/kernel*
dtype0*(
_output_shapes
:

conv2d_3/bias/Initializer/zerosConst* 
_class
loc:@conv2d_3/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ś
conv2d_3/biasVarHandleOp*
_output_shapes
: *
shared_nameconv2d_3/bias* 
_class
loc:@conv2d_3/bias*
	container *
shape:*
dtype0
k
.conv2d_3/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_3/bias*
_output_shapes
: 

conv2d_3/bias/AssignAssignVariableOpconv2d_3/biasconv2d_3/bias/Initializer/zeros*
dtype0* 
_class
loc:@conv2d_3/bias

!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias* 
_class
loc:@conv2d_3/bias*
dtype0*
_output_shapes	
:
g
conv2d_3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
x
conv2d_3/Conv2D/ReadVariableOpReadVariableOpconv2d_3/kernel*
dtype0*(
_output_shapes
:
ň
conv2d_3/Conv2DConv2Dconv2d_2/Reluconv2d_3/Conv2D/ReadVariableOp*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	dilations
*
T0
j
conv2d_3/BiasAdd/ReadVariableOpReadVariableOpconv2d_3/bias*
dtype0*
_output_shapes	
:

conv2d_3/BiasAddBiasAddconv2d_3/Conv2Dconv2d_3/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
conv2d_3/ReluReluconv2d_3/BiasAdd*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
h
dropout_1/IdentityIdentityconv2d_3/Relu*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
­
0conv2d_4/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@conv2d_4/kernel*%
valueB"            *
dtype0*
_output_shapes
:

.conv2d_4/kernel/Initializer/random_uniform/minConst*"
_class
loc:@conv2d_4/kernel*
valueB
 *   ˝*
dtype0*
_output_shapes
: 

.conv2d_4/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@conv2d_4/kernel*
valueB
 *   =*
dtype0*
_output_shapes
: 
ř
8conv2d_4/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_4/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0*(
_output_shapes
:*

seed *
T0*"
_class
loc:@conv2d_4/kernel
Ú
.conv2d_4/kernel/Initializer/random_uniform/subSub.conv2d_4/kernel/Initializer/random_uniform/max.conv2d_4/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_4/kernel*
_output_shapes
: *
T0
ö
.conv2d_4/kernel/Initializer/random_uniform/mulMul8conv2d_4/kernel/Initializer/random_uniform/RandomUniform.conv2d_4/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv2d_4/kernel*(
_output_shapes
:
č
*conv2d_4/kernel/Initializer/random_uniformAdd.conv2d_4/kernel/Initializer/random_uniform/mul.conv2d_4/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_4/kernel*(
_output_shapes
:
š
conv2d_4/kernelVarHandleOp*
shape:*
dtype0*
_output_shapes
: * 
shared_nameconv2d_4/kernel*"
_class
loc:@conv2d_4/kernel*
	container 
o
0conv2d_4/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_4/kernel*
_output_shapes
: 

conv2d_4/kernel/AssignAssignVariableOpconv2d_4/kernel*conv2d_4/kernel/Initializer/random_uniform*"
_class
loc:@conv2d_4/kernel*
dtype0
Ą
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*"
_class
loc:@conv2d_4/kernel*
dtype0*(
_output_shapes
:

conv2d_4/bias/Initializer/zerosConst* 
_class
loc:@conv2d_4/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ś
conv2d_4/biasVarHandleOp* 
_class
loc:@conv2d_4/bias*
	container *
shape:*
dtype0*
_output_shapes
: *
shared_nameconv2d_4/bias
k
.conv2d_4/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_4/bias*
_output_shapes
: 

conv2d_4/bias/AssignAssignVariableOpconv2d_4/biasconv2d_4/bias/Initializer/zeros* 
_class
loc:@conv2d_4/bias*
dtype0

!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
dtype0*
_output_shapes	
:* 
_class
loc:@conv2d_4/bias
g
conv2d_4/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
x
conv2d_4/Conv2D/ReadVariableOpReadVariableOpconv2d_4/kernel*
dtype0*(
_output_shapes
:
÷
conv2d_4/Conv2DConv2Ddropout_1/Identityconv2d_4/Conv2D/ReadVariableOp*
paddingVALID*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
j
conv2d_4/BiasAdd/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes	
:*
dtype0

conv2d_4/BiasAddBiasAddconv2d_4/Conv2Dconv2d_4/BiasAdd/ReadVariableOp*
data_formatNHWC*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		*
T0
b
conv2d_4/ReluReluconv2d_4/BiasAdd*0
_output_shapes
:˙˙˙˙˙˙˙˙˙		*
T0
­
0conv2d_5/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*"
_class
loc:@conv2d_5/kernel*%
valueB"            

.conv2d_5/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *"
_class
loc:@conv2d_5/kernel*
valueB
 *×łÝź

.conv2d_5/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@conv2d_5/kernel*
valueB
 *×łÝ<*
dtype0*
_output_shapes
: 
ř
8conv2d_5/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_5/kernel/Initializer/random_uniform/shape*"
_class
loc:@conv2d_5/kernel*
seed2 *
dtype0*(
_output_shapes
:*

seed *
T0
Ú
.conv2d_5/kernel/Initializer/random_uniform/subSub.conv2d_5/kernel/Initializer/random_uniform/max.conv2d_5/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_5/kernel*
_output_shapes
: 
ö
.conv2d_5/kernel/Initializer/random_uniform/mulMul8conv2d_5/kernel/Initializer/random_uniform/RandomUniform.conv2d_5/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv2d_5/kernel*(
_output_shapes
:
č
*conv2d_5/kernel/Initializer/random_uniformAdd.conv2d_5/kernel/Initializer/random_uniform/mul.conv2d_5/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_5/kernel*(
_output_shapes
:
š
conv2d_5/kernelVarHandleOp* 
shared_nameconv2d_5/kernel*"
_class
loc:@conv2d_5/kernel*
	container *
shape:*
dtype0*
_output_shapes
: 
o
0conv2d_5/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_5/kernel*
_output_shapes
: 

conv2d_5/kernel/AssignAssignVariableOpconv2d_5/kernel*conv2d_5/kernel/Initializer/random_uniform*"
_class
loc:@conv2d_5/kernel*
dtype0
Ą
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*"
_class
loc:@conv2d_5/kernel*
dtype0*(
_output_shapes
:

conv2d_5/bias/Initializer/zerosConst* 
_class
loc:@conv2d_5/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ś
conv2d_5/biasVarHandleOp*
shape:*
dtype0*
_output_shapes
: *
shared_nameconv2d_5/bias* 
_class
loc:@conv2d_5/bias*
	container 
k
.conv2d_5/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_5/bias*
_output_shapes
: 

conv2d_5/bias/AssignAssignVariableOpconv2d_5/biasconv2d_5/bias/Initializer/zeros* 
_class
loc:@conv2d_5/bias*
dtype0

!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias* 
_class
loc:@conv2d_5/bias*
dtype0*
_output_shapes	
:
g
conv2d_5/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
x
conv2d_5/Conv2D/ReadVariableOpReadVariableOpconv2d_5/kernel*
dtype0*(
_output_shapes
:
ň
conv2d_5/Conv2DConv2Dconv2d_4/Reluconv2d_5/Conv2D/ReadVariableOp*
paddingVALID*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
j
conv2d_5/BiasAdd/ReadVariableOpReadVariableOpconv2d_5/bias*
dtype0*
_output_shapes	
:

conv2d_5/BiasAddBiasAddconv2d_5/Conv2Dconv2d_5/BiasAdd/ReadVariableOp*
data_formatNHWC*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
b
conv2d_5/ReluReluconv2d_5/BiasAdd*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙
Z
flatten/ShapeShapeconv2d_5/Relu*
_output_shapes
:*
T0*
out_type0
e
flatten/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0
g
flatten/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
g
flatten/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ą
flatten/strided_sliceStridedSliceflatten/Shapeflatten/strided_slice/stackflatten/strided_slice/stack_1flatten/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
b
flatten/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙

flatten/Reshape/shapePackflatten/strided_sliceflatten/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:

flatten/ReshapeReshapeconv2d_5/Reluflatten/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
b
dropout_2/IdentityIdentityflatten/Reshape*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

-dense/kernel/Initializer/random_uniform/shapeConst*
_class
loc:@dense/kernel*
valueB" 	     *
dtype0*
_output_shapes
:

+dense/kernel/Initializer/random_uniform/minConst*
_class
loc:@dense/kernel*
valueB
 *=˝*
dtype0*
_output_shapes
: 

+dense/kernel/Initializer/random_uniform/maxConst*
_class
loc:@dense/kernel*
valueB
 *==*
dtype0*
_output_shapes
: 
ç
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
T0*
_class
loc:@dense/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed 
Î
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*
_class
loc:@dense/kernel
â
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@dense/kernel* 
_output_shapes
:

Ô
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel* 
_output_shapes
:

¨
dense/kernelVarHandleOp*
_class
loc:@dense/kernel*
	container *
shape:
*
dtype0*
_output_shapes
: *
shared_namedense/kernel
i
-dense/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense/kernel*
_output_shapes
: 

dense/kernel/AssignAssignVariableOpdense/kernel'dense/kernel/Initializer/random_uniform*
_class
loc:@dense/kernel*
dtype0

 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
*
_class
loc:@dense/kernel*
dtype0

dense/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
_class
loc:@dense/bias*
valueB*    


dense/biasVarHandleOp*
_class
loc:@dense/bias*
	container *
shape:*
dtype0*
_output_shapes
: *
shared_name
dense/bias
e
+dense/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp
dense/bias*
_output_shapes
: 
{
dense/bias/AssignAssignVariableOp
dense/biasdense/bias/Initializer/zeros*
_class
loc:@dense/bias*
dtype0

dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_class
loc:@dense/bias*
dtype0*
_output_shapes	
:
j
dense/MatMul/ReadVariableOpReadVariableOpdense/kernel*
dtype0* 
_output_shapes
:

 
dense/MatMulMatMuldropout_2/Identitydense/MatMul/ReadVariableOp*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0
d
dense/BiasAdd/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes	
:

dense/BiasAddBiasAdddense/MatMuldense/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
T

dense/ReluReludense/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
/dense_1/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*!
_class
loc:@dense_1/kernel*
valueB"      *
dtype0

-dense_1/kernel/Initializer/random_uniform/minConst*!
_class
loc:@dense_1/kernel*
valueB
 *ľ­×˝*
dtype0*
_output_shapes
: 

-dense_1/kernel/Initializer/random_uniform/maxConst*!
_class
loc:@dense_1/kernel*
valueB
 *ľ­×=*
dtype0*
_output_shapes
: 
ě
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*
T0*!
_class
loc:@dense_1/kernel*
seed2 *
dtype0*
_output_shapes
:	*

seed 
Ö
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
: 
é
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*!
_class
loc:@dense_1/kernel*
_output_shapes
:	*
T0
Ű
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
:	
­
dense_1/kernelVarHandleOp*
	container *
shape:	*
dtype0*
_output_shapes
: *
shared_namedense_1/kernel*!
_class
loc:@dense_1/kernel
m
/dense_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/kernel*
_output_shapes
: 

dense_1/kernel/AssignAssignVariableOpdense_1/kernel)dense_1/kernel/Initializer/random_uniform*!
_class
loc:@dense_1/kernel*
dtype0

"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
:	

dense_1/bias/Initializer/zerosConst*
_output_shapes
:*
_class
loc:@dense_1/bias*
valueB*    *
dtype0
˘
dense_1/biasVarHandleOp*
_class
loc:@dense_1/bias*
	container *
shape:*
dtype0*
_output_shapes
: *
shared_namedense_1/bias
i
-dense_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/bias*
_output_shapes
: 

dense_1/bias/AssignAssignVariableOpdense_1/biasdense_1/bias/Initializer/zeros*
_class
loc:@dense_1/bias*
dtype0

 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
_class
loc:@dense_1/bias*
dtype0
m
dense_1/MatMul/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*
_output_shapes
:	

dense_1/MatMulMatMul
dense/Reludense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 
g
dense_1/BiasAdd/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes
:

dense_1/BiasAddBiasAdddense_1/MatMuldense_1/BiasAdd/ReadVariableOp*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC
]
dense_1/SoftmaxSoftmaxdense_1/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
$

group_depsNoOp^dense_1/Softmax
U
ConstConst"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_1Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_2Const"/device:CPU:0*
_output_shapes
: *
valueB B *
dtype0
W
Const_3Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_4Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_5Const"/device:CPU:0*
_output_shapes
: *
valueB B *
dtype0
W
Const_6Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_7Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_8Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
W
Const_9Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_10Const"/device:CPU:0*
_output_shapes
: *
valueB B *
dtype0
X
Const_11Const"/device:CPU:0*
dtype0*
_output_shapes
: *
valueB B 
X
Const_12Const"/device:CPU:0*
dtype0*
_output_shapes
: *
valueB B 
X
Const_13Const"/device:CPU:0*
dtype0*
_output_shapes
: *
valueB B 
]
Const_14Const"/device:CPU:0*
valueB Bmodel*
dtype0*
_output_shapes
: 
X
Const_15Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_16Const"/device:CPU:0*
dtype0*
_output_shapes
: *
valueB B 
X
Const_17Const"/device:CPU:0*
_output_shapes
: *
valueB B *
dtype0
X
Const_18Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_19Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_20Const"/device:CPU:0*
dtype0*
_output_shapes
: *
valueB B 
X
Const_21Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_22Const"/device:CPU:0*
_output_shapes
: *
valueB B *
dtype0
X
Const_23Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_24Const"/device:CPU:0*
dtype0*
_output_shapes
: *
valueB B 
X
Const_25Const"/device:CPU:0*
dtype0*
_output_shapes
: *
valueB B 
X
Const_26Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_27Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
X
Const_28Const"/device:CPU:0*
_output_shapes
: *
valueB B *
dtype0
¤
RestoreV2/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
r
RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

	RestoreV2	RestoreV2Const_14RestoreV2/tensor_namesRestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
B
IdentityIdentity	RestoreV2*
_output_shapes
:*
T0
J
AssignVariableOpAssignVariableOpconv2d/kernelIdentity*
dtype0
¤
RestoreV2_1/tensor_namesConst"/device:CPU:0*I
value@B>B4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_1	RestoreV2Const_14RestoreV2_1/tensor_namesRestoreV2_1/shape_and_slices"/device:CPU:0*
_output_shapes
:*
dtypes
2
F

Identity_1IdentityRestoreV2_1*
T0*
_output_shapes
:
L
AssignVariableOp_1AssignVariableOpconv2d/bias
Identity_1*
dtype0
Ś
RestoreV2_2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*K
valueBB@B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0
t
RestoreV2_2/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_2	RestoreV2Const_14RestoreV2_2/tensor_namesRestoreV2_2/shape_and_slices"/device:CPU:0*
_output_shapes
:*
dtypes
2
F

Identity_2IdentityRestoreV2_2*
T0*
_output_shapes
:
P
AssignVariableOp_2AssignVariableOpconv2d_1/kernel
Identity_2*
dtype0
¤
RestoreV2_3/tensor_namesConst"/device:CPU:0*I
value@B>B4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
t
RestoreV2_3/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_3	RestoreV2Const_14RestoreV2_3/tensor_namesRestoreV2_3/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
F

Identity_3IdentityRestoreV2_3*
_output_shapes
:*
T0
N
AssignVariableOp_3AssignVariableOpconv2d_1/bias
Identity_3*
dtype0
Ś
RestoreV2_4/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
t
RestoreV2_4/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB
B 

RestoreV2_4	RestoreV2Const_14RestoreV2_4/tensor_namesRestoreV2_4/shape_and_slices"/device:CPU:0*
_output_shapes
:*
dtypes
2
F

Identity_4IdentityRestoreV2_4*
T0*
_output_shapes
:
P
AssignVariableOp_4AssignVariableOpconv2d_2/kernel
Identity_4*
dtype0
¤
RestoreV2_5/tensor_namesConst"/device:CPU:0*I
value@B>B4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
t
RestoreV2_5/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_5	RestoreV2Const_14RestoreV2_5/tensor_namesRestoreV2_5/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
F

Identity_5IdentityRestoreV2_5*
T0*
_output_shapes
:
N
AssignVariableOp_5AssignVariableOpconv2d_2/bias
Identity_5*
dtype0
Ś
RestoreV2_6/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
t
RestoreV2_6/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_6	RestoreV2Const_14RestoreV2_6/tensor_namesRestoreV2_6/shape_and_slices"/device:CPU:0*
_output_shapes
:*
dtypes
2
F

Identity_6IdentityRestoreV2_6*
T0*
_output_shapes
:
P
AssignVariableOp_6AssignVariableOpconv2d_3/kernel
Identity_6*
dtype0
¤
RestoreV2_7/tensor_namesConst"/device:CPU:0*I
value@B>B4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
t
RestoreV2_7/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_7	RestoreV2Const_14RestoreV2_7/tensor_namesRestoreV2_7/shape_and_slices"/device:CPU:0*
_output_shapes
:*
dtypes
2
F

Identity_7IdentityRestoreV2_7*
T0*
_output_shapes
:
N
AssignVariableOp_7AssignVariableOpconv2d_3/bias
Identity_7*
dtype0
Ś
RestoreV2_8/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
t
RestoreV2_8/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_8	RestoreV2Const_14RestoreV2_8/tensor_namesRestoreV2_8/shape_and_slices"/device:CPU:0*
_output_shapes
:*
dtypes
2
F

Identity_8IdentityRestoreV2_8*
T0*
_output_shapes
:
P
AssignVariableOp_8AssignVariableOpconv2d_4/kernel
Identity_8*
dtype0
¤
RestoreV2_9/tensor_namesConst"/device:CPU:0*I
value@B>B4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
t
RestoreV2_9/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB
B 

RestoreV2_9	RestoreV2Const_14RestoreV2_9/tensor_namesRestoreV2_9/shape_and_slices"/device:CPU:0*
_output_shapes
:*
dtypes
2
F

Identity_9IdentityRestoreV2_9*
_output_shapes
:*
T0
N
AssignVariableOp_9AssignVariableOpconv2d_4/bias
Identity_9*
dtype0
§
RestoreV2_10/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_10/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
valueB
B *
dtype0

RestoreV2_10	RestoreV2Const_14RestoreV2_10/tensor_namesRestoreV2_10/shape_and_slices"/device:CPU:0*
_output_shapes
:*
dtypes
2
H
Identity_10IdentityRestoreV2_10*
T0*
_output_shapes
:
R
AssignVariableOp_10AssignVariableOpconv2d_5/kernelIdentity_10*
dtype0
Ľ
RestoreV2_11/tensor_namesConst"/device:CPU:0*I
value@B>B4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_11/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_11	RestoreV2Const_14RestoreV2_11/tensor_namesRestoreV2_11/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_11IdentityRestoreV2_11*
T0*
_output_shapes
:
P
AssignVariableOp_11AssignVariableOpconv2d_5/biasIdentity_11*
dtype0
§
RestoreV2_12/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_12/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_12	RestoreV2Const_14RestoreV2_12/tensor_namesRestoreV2_12/shape_and_slices"/device:CPU:0*
_output_shapes
:*
dtypes
2
H
Identity_12IdentityRestoreV2_12*
T0*
_output_shapes
:
O
AssignVariableOp_12AssignVariableOpdense/kernelIdentity_12*
dtype0
Ľ
RestoreV2_13/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*I
value@B>B4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
u
RestoreV2_13/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_13	RestoreV2Const_14RestoreV2_13/tensor_namesRestoreV2_13/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_13IdentityRestoreV2_13*
_output_shapes
:*
T0
M
AssignVariableOp_13AssignVariableOp
dense/biasIdentity_13*
dtype0
§
RestoreV2_14/tensor_namesConst"/device:CPU:0*K
valueBB@B6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_14/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB
B 

RestoreV2_14	RestoreV2Const_14RestoreV2_14/tensor_namesRestoreV2_14/shape_and_slices"/device:CPU:0*
_output_shapes
:*
dtypes
2
H
Identity_14IdentityRestoreV2_14*
T0*
_output_shapes
:
Q
AssignVariableOp_14AssignVariableOpdense_1/kernelIdentity_14*
dtype0
Ľ
RestoreV2_15/tensor_namesConst"/device:CPU:0*I
value@B>B4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
u
RestoreV2_15/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:

RestoreV2_15	RestoreV2Const_14RestoreV2_15/tensor_namesRestoreV2_15/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
H
Identity_15IdentityRestoreV2_15*
T0*
_output_shapes
:
O
AssignVariableOp_15AssignVariableOpdense_1/biasIdentity_15*
dtype0
Q
VarIsInitializedOpVarIsInitializedOpconv2d_1/kernel*
_output_shapes
: 
S
VarIsInitializedOp_1VarIsInitializedOpconv2d_5/kernel*
_output_shapes
: 
P
VarIsInitializedOp_2VarIsInitializedOpdense_1/bias*
_output_shapes
: 
S
VarIsInitializedOp_3VarIsInitializedOpconv2d_3/kernel*
_output_shapes
: 
Q
VarIsInitializedOp_4VarIsInitializedOpconv2d_4/bias*
_output_shapes
: 
R
VarIsInitializedOp_5VarIsInitializedOpdense_1/kernel*
_output_shapes
: 
S
VarIsInitializedOp_6VarIsInitializedOpconv2d_4/kernel*
_output_shapes
: 
Q
VarIsInitializedOp_7VarIsInitializedOpconv2d_3/bias*
_output_shapes
: 
Q
VarIsInitializedOp_8VarIsInitializedOpconv2d_5/bias*
_output_shapes
: 
N
VarIsInitializedOp_9VarIsInitializedOp
dense/bias*
_output_shapes
: 
T
VarIsInitializedOp_10VarIsInitializedOpconv2d_2/kernel*
_output_shapes
: 
R
VarIsInitializedOp_11VarIsInitializedOpconv2d_1/bias*
_output_shapes
: 
R
VarIsInitializedOp_12VarIsInitializedOpconv2d/kernel*
_output_shapes
: 
Q
VarIsInitializedOp_13VarIsInitializedOpdense/kernel*
_output_shapes
: 
P
VarIsInitializedOp_14VarIsInitializedOpconv2d/bias*
_output_shapes
: 
R
VarIsInitializedOp_15VarIsInitializedOpconv2d_2/bias*
_output_shapes
: 

initNoOp^conv2d/bias/Assign^conv2d/kernel/Assign^conv2d_1/bias/Assign^conv2d_1/kernel/Assign^conv2d_2/bias/Assign^conv2d_2/kernel/Assign^conv2d_3/bias/Assign^conv2d_3/kernel/Assign^conv2d_4/bias/Assign^conv2d_4/kernel/Assign^conv2d_5/bias/Assign^conv2d_5/kernel/Assign^dense/bias/Assign^dense/kernel/Assign^dense_1/bias/Assign^dense_1/kernel/Assign
X
Const_29Const"/device:CPU:0*
valueB B *
dtype0*
_output_shapes
: 
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save/SaveV2/tensor_namesConst*ľ
valueŤB¨B/.ATTRIBUTES/OBJECT_CONFIG_JSONB_CHECKPOINTABLE_OBJECT_GRAPHB&layer-0/.ATTRIBUTES/OBJECT_CONFIG_JSONB'layer-10/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-3/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-6/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-9/.ATTRIBUTES/OBJECT_CONFIG_JSONB3layer_with_weights-0/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-1/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-2/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-3/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-4/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-5/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-6/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-7/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
Ą
save/SaveV2/shape_and_slicesConst*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
Ň
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesConst_15Const_29Const_16Const_26Const_19Const_22Const_25Const_17conv2d/bias/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpConst_18!conv2d_1/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOpConst_20!conv2d_2/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOpConst_21!conv2d_3/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOpConst_23!conv2d_4/bias/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOpConst_24!conv2d_5/bias/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOpConst_27dense/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpConst_28 dense_1/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp*-
dtypes#
!2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 

save/RestoreV2/tensor_namesConst"/device:CPU:0*ľ
valueŤB¨B/.ATTRIBUTES/OBJECT_CONFIG_JSONB_CHECKPOINTABLE_OBJECT_GRAPHB&layer-0/.ATTRIBUTES/OBJECT_CONFIG_JSONB'layer-10/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-3/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-6/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-9/.ATTRIBUTES/OBJECT_CONFIG_JSONB3layer_with_weights-0/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-1/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-2/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-3/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-4/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-5/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-6/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-7/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
ł
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
ś
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
_output_shapes~
|:::::::::::::::::::::::::::::::*-
dtypes#
!2

	save/NoOpNoOp

save/NoOp_1NoOp

save/NoOp_2NoOp

save/NoOp_3NoOp

save/NoOp_4NoOp

save/NoOp_5NoOp

save/NoOp_6NoOp

save/NoOp_7NoOp
N
save/IdentityIdentitysave/RestoreV2:8*
T0*
_output_shapes
:
R
save/AssignVariableOpAssignVariableOpconv2d/biassave/Identity*
dtype0
P
save/Identity_1Identitysave/RestoreV2:9*
_output_shapes
:*
T0
X
save/AssignVariableOp_1AssignVariableOpconv2d/kernelsave/Identity_1*
dtype0

save/NoOp_8NoOp
Q
save/Identity_2Identitysave/RestoreV2:11*
_output_shapes
:*
T0
X
save/AssignVariableOp_2AssignVariableOpconv2d_1/biassave/Identity_2*
dtype0
Q
save/Identity_3Identitysave/RestoreV2:12*
T0*
_output_shapes
:
Z
save/AssignVariableOp_3AssignVariableOpconv2d_1/kernelsave/Identity_3*
dtype0

save/NoOp_9NoOp
Q
save/Identity_4Identitysave/RestoreV2:14*
T0*
_output_shapes
:
X
save/AssignVariableOp_4AssignVariableOpconv2d_2/biassave/Identity_4*
dtype0
Q
save/Identity_5Identitysave/RestoreV2:15*
T0*
_output_shapes
:
Z
save/AssignVariableOp_5AssignVariableOpconv2d_2/kernelsave/Identity_5*
dtype0

save/NoOp_10NoOp
Q
save/Identity_6Identitysave/RestoreV2:17*
_output_shapes
:*
T0
X
save/AssignVariableOp_6AssignVariableOpconv2d_3/biassave/Identity_6*
dtype0
Q
save/Identity_7Identitysave/RestoreV2:18*
T0*
_output_shapes
:
Z
save/AssignVariableOp_7AssignVariableOpconv2d_3/kernelsave/Identity_7*
dtype0

save/NoOp_11NoOp
Q
save/Identity_8Identitysave/RestoreV2:20*
T0*
_output_shapes
:
X
save/AssignVariableOp_8AssignVariableOpconv2d_4/biassave/Identity_8*
dtype0
Q
save/Identity_9Identitysave/RestoreV2:21*
T0*
_output_shapes
:
Z
save/AssignVariableOp_9AssignVariableOpconv2d_4/kernelsave/Identity_9*
dtype0

save/NoOp_12NoOp
R
save/Identity_10Identitysave/RestoreV2:23*
_output_shapes
:*
T0
Z
save/AssignVariableOp_10AssignVariableOpconv2d_5/biassave/Identity_10*
dtype0
R
save/Identity_11Identitysave/RestoreV2:24*
_output_shapes
:*
T0
\
save/AssignVariableOp_11AssignVariableOpconv2d_5/kernelsave/Identity_11*
dtype0

save/NoOp_13NoOp
R
save/Identity_12Identitysave/RestoreV2:26*
T0*
_output_shapes
:
W
save/AssignVariableOp_12AssignVariableOp
dense/biassave/Identity_12*
dtype0
R
save/Identity_13Identitysave/RestoreV2:27*
T0*
_output_shapes
:
Y
save/AssignVariableOp_13AssignVariableOpdense/kernelsave/Identity_13*
dtype0

save/NoOp_14NoOp
R
save/Identity_14Identitysave/RestoreV2:29*
_output_shapes
:*
T0
Y
save/AssignVariableOp_14AssignVariableOpdense_1/biassave/Identity_14*
dtype0
R
save/Identity_15Identitysave/RestoreV2:30*
T0*
_output_shapes
:
[
save/AssignVariableOp_15AssignVariableOpdense_1/kernelsave/Identity_15*
dtype0

save/restore_allNoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_10^save/AssignVariableOp_11^save/AssignVariableOp_12^save/AssignVariableOp_13^save/AssignVariableOp_14^save/AssignVariableOp_15^save/AssignVariableOp_2^save/AssignVariableOp_3^save/AssignVariableOp_4^save/AssignVariableOp_5^save/AssignVariableOp_6^save/AssignVariableOp_7^save/AssignVariableOp_8^save/AssignVariableOp_9
^save/NoOp^save/NoOp_1^save/NoOp_10^save/NoOp_11^save/NoOp_12^save/NoOp_13^save/NoOp_14^save/NoOp_2^save/NoOp_3^save/NoOp_4^save/NoOp_5^save/NoOp_6^save/NoOp_7^save/NoOp_8^save/NoOp_9
R
save_1/ConstConst*
_output_shapes
: *
valueB Bmodel*
dtype0
ć
save_1/SaveV2/tensor_namesConst*
valueBB/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-0/.ATTRIBUTES/OBJECT_CONFIG_JSONB'layer-10/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-3/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-6/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-9/.ATTRIBUTES/OBJECT_CONFIG_JSONB3layer_with_weights-0/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-1/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-2/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-3/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-4/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-5/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-6/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-7/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
Ą
save_1/SaveV2/shape_and_slicesConst*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
Ü(
save_1/SaveV2/tensors_0Const*(
value(B( B({"class_name": "Sequential", "config": {"layers": [{"class_name": "Conv2D", "config": {"activation": "relu", "activity_regularizer": null, "batch_input_shape": [null, 64, 64, 3], "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_regularizer": null, "data_format": "channels_last", "dilation_rate": [1, 1], "dtype": "float32", "filters": 64, "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"dtype": "float32", "seed": null}}, "kernel_regularizer": null, "kernel_size": [4, 4], "name": "conv2d", "padding": "valid", "strides": [1, 1], "trainable": true, "use_bias": true}}, {"class_name": "Conv2D", "config": {"activation": "relu", "activity_regularizer": null, "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_regularizer": null, "data_format": "channels_last", "dilation_rate": [1, 1], "dtype": "float32", "filters": 64, "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"dtype": "float32", "seed": null}}, "kernel_regularizer": null, "kernel_size": [4, 4], "name": "conv2d_1", "padding": "valid", "strides": [2, 2], "trainable": true, "use_bias": true}}, {"class_name": "Dropout", "config": {"dtype": "float32", "name": "dropout", "noise_shape": null, "rate": 0.5, "seed": null, "trainable": true}}, {"class_name": "Conv2D", "config": {"activation": "relu", "activity_regularizer": null, "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_regularizer": null, "data_format": "channels_last", "dilation_rate": [1, 1], "dtype": "float32", "filters": 128, "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"dtype": "float32", "seed": null}}, "kernel_regularizer": null, "kernel_size": [4, 4], "name": "conv2d_2", "padding": "valid", "strides": [1, 1], "trainable": true, "use_bias": true}}, {"class_name": "Conv2D", "config": {"activation": "relu", "activity_regularizer": null, "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_regularizer": null, "data_format": "channels_last", "dilation_rate": [1, 1], "dtype": "float32", "filters": 128, "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"dtype": "float32", "seed": null}}, "kernel_regularizer": null, "kernel_size": [4, 4], "name": "conv2d_3", "padding": "valid", "strides": [2, 2], "trainable": true, "use_bias": true}}, {"class_name": "Dropout", "config": {"dtype": "float32", "name": "dropout_1", "noise_shape": null, "rate": 0.5, "seed": null, "trainable": true}}, {"class_name": "Conv2D", "config": {"activation": "relu", "activity_regularizer": null, "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_regularizer": null, "data_format": "channels_last", "dilation_rate": [1, 1], "dtype": "float32", "filters": 256, "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"dtype": "float32", "seed": null}}, "kernel_regularizer": null, "kernel_size": [4, 4], "name": "conv2d_4", "padding": "valid", "strides": [1, 1], "trainable": true, "use_bias": true}}, {"class_name": "Conv2D", "config": {"activation": "relu", "activity_regularizer": null, "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_regularizer": null, "data_format": "channels_last", "dilation_rate": [1, 1], "dtype": "float32", "filters": 256, "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"dtype": "float32", "seed": null}}, "kernel_regularizer": null, "kernel_size": [4, 4], "name": "conv2d_5", "padding": "valid", "strides": [2, 2], "trainable": true, "use_bias": true}}, {"class_name": "Flatten", "config": {"data_format": "channels_last", "dtype": "float32", "name": "flatten", "trainable": true}}, {"class_name": "Dropout", "config": {"dtype": "float32", "name": "dropout_2", "noise_shape": null, "rate": 0.5, "seed": null, "trainable": true}}, {"class_name": "Dense", "config": {"activation": "relu", "activity_regularizer": null, "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_regularizer": null, "dtype": "float32", "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"dtype": "float32", "seed": null}}, "kernel_regularizer": null, "name": "dense", "trainable": true, "units": 512, "use_bias": true}}, {"class_name": "Dense", "config": {"activation": "softmax", "activity_regularizer": null, "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_regularizer": null, "dtype": "float32", "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"dtype": "float32", "seed": null}}, "kernel_regularizer": null, "name": "dense_1", "trainable": true, "units": 29, "use_bias": true}}], "name": "sequential"}}*
dtype0*
_output_shapes
: 
é
save_1/SaveV2/tensors_1Const*Ą
valueB B{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 64, 64, 3], "dtype": "float32", "name": "conv2d_input", "sparse": false}}*
dtype0*
_output_shapes
: 
í
save_1/SaveV2/tensors_2Const*Ľ
valueB B{"class_name": "Dropout", "config": {"dtype": "float32", "name": "dropout_2", "noise_shape": null, "rate": 0.5, "seed": null, "trainable": true}}*
dtype0*
_output_shapes
: 
ë
save_1/SaveV2/tensors_3Const*Ł
valueB B{"class_name": "Dropout", "config": {"dtype": "float32", "name": "dropout", "noise_shape": null, "rate": 0.5, "seed": null, "trainable": true}}*
dtype0*
_output_shapes
: 
í
save_1/SaveV2/tensors_4Const*Ľ
valueB B{"class_name": "Dropout", "config": {"dtype": "float32", "name": "dropout_1", "noise_shape": null, "rate": 0.5, "seed": null, "trainable": true}}*
dtype0*
_output_shapes
: 
Ú
save_1/SaveV2/tensors_5Const*
dtype0*
_output_shapes
: *
valueB B{"class_name": "Flatten", "config": {"data_format": "channels_last", "dtype": "float32", "name": "flatten", "trainable": true}}
É
save_1/SaveV2/tensors_6Const*
value÷Bô Bí{"class_name": "Conv2D", "config": {"activation": "relu", "activity_regularizer": null, "batch_input_shape": [null, 64, 64, 3], "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_regularizer": null, "data_format": "channels_last", "dilation_rate": [1, 1], "dtype": "float32", "filters": 64, "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"dtype": "float32", "seed": null}}, "kernel_regularizer": null, "kernel_size": [4, 4], "name": "conv2d", "padding": "valid", "strides": [1, 1], "trainable": true, "use_bias": true}}*
dtype0*
_output_shapes
: 
Ł
save_1/SaveV2/tensors_9Const*Ű
valueŃBÎ BÇ{"class_name": "Conv2D", "config": {"activation": "relu", "activity_regularizer": null, "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_regularizer": null, "data_format": "channels_last", "dilation_rate": [1, 1], "dtype": "float32", "filters": 64, "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"dtype": "float32", "seed": null}}, "kernel_regularizer": null, "kernel_size": [4, 4], "name": "conv2d_1", "padding": "valid", "strides": [2, 2], "trainable": true, "use_bias": true}}*
dtype0*
_output_shapes
: 
Ľ
save_1/SaveV2/tensors_12Const*Ü
valueŇBĎ BČ{"class_name": "Conv2D", "config": {"activation": "relu", "activity_regularizer": null, "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_regularizer": null, "data_format": "channels_last", "dilation_rate": [1, 1], "dtype": "float32", "filters": 128, "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"dtype": "float32", "seed": null}}, "kernel_regularizer": null, "kernel_size": [4, 4], "name": "conv2d_2", "padding": "valid", "strides": [1, 1], "trainable": true, "use_bias": true}}*
dtype0*
_output_shapes
: 
Ľ
save_1/SaveV2/tensors_15Const*Ü
valueŇBĎ BČ{"class_name": "Conv2D", "config": {"activation": "relu", "activity_regularizer": null, "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_regularizer": null, "data_format": "channels_last", "dilation_rate": [1, 1], "dtype": "float32", "filters": 128, "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"dtype": "float32", "seed": null}}, "kernel_regularizer": null, "kernel_size": [4, 4], "name": "conv2d_3", "padding": "valid", "strides": [2, 2], "trainable": true, "use_bias": true}}*
dtype0*
_output_shapes
: 
Ľ
save_1/SaveV2/tensors_18Const*
_output_shapes
: *Ü
valueŇBĎ BČ{"class_name": "Conv2D", "config": {"activation": "relu", "activity_regularizer": null, "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_regularizer": null, "data_format": "channels_last", "dilation_rate": [1, 1], "dtype": "float32", "filters": 256, "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"dtype": "float32", "seed": null}}, "kernel_regularizer": null, "kernel_size": [4, 4], "name": "conv2d_4", "padding": "valid", "strides": [1, 1], "trainable": true, "use_bias": true}}*
dtype0
Ľ
save_1/SaveV2/tensors_21Const*Ü
valueŇBĎ BČ{"class_name": "Conv2D", "config": {"activation": "relu", "activity_regularizer": null, "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_regularizer": null, "data_format": "channels_last", "dilation_rate": [1, 1], "dtype": "float32", "filters": 256, "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"dtype": "float32", "seed": null}}, "kernel_regularizer": null, "kernel_size": [4, 4], "name": "conv2d_5", "padding": "valid", "strides": [2, 2], "trainable": true, "use_bias": true}}*
dtype0*
_output_shapes
: 
¨
save_1/SaveV2/tensors_24Const*ß
valueŐBŇ BË{"class_name": "Dense", "config": {"activation": "relu", "activity_regularizer": null, "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_regularizer": null, "dtype": "float32", "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"dtype": "float32", "seed": null}}, "kernel_regularizer": null, "name": "dense", "trainable": true, "units": 512, "use_bias": true}}*
dtype0*
_output_shapes
: 
Ź
save_1/SaveV2/tensors_27Const*ă
valueŮBÖ BĎ{"class_name": "Dense", "config": {"activation": "softmax", "activity_regularizer": null, "bias_constraint": null, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "bias_regularizer": null, "dtype": "float32", "kernel_constraint": null, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"dtype": "float32", "seed": null}}, "kernel_regularizer": null, "name": "dense_1", "trainable": true, "units": 29, "use_bias": true}}*
dtype0*
_output_shapes
: 
§
save_1/SaveV2SaveV2save_1/Constsave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicessave_1/SaveV2/tensors_0save_1/SaveV2/tensors_1save_1/SaveV2/tensors_2save_1/SaveV2/tensors_3save_1/SaveV2/tensors_4save_1/SaveV2/tensors_5save_1/SaveV2/tensors_6conv2d/bias/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpsave_1/SaveV2/tensors_9!conv2d_1/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOpsave_1/SaveV2/tensors_12!conv2d_2/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOpsave_1/SaveV2/tensors_15!conv2d_3/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOpsave_1/SaveV2/tensors_18!conv2d_4/bias/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOpsave_1/SaveV2/tensors_21!conv2d_5/bias/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOpsave_1/SaveV2/tensors_24dense/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpsave_1/SaveV2/tensors_27 dense_1/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp*,
dtypes"
 2

save_1/control_dependencyIdentitysave_1/Const^save_1/SaveV2*
T0*
_class
loc:@save_1/Const*
_output_shapes
: 
ř
save_1/RestoreV2/tensor_namesConst"/device:CPU:0*
valueBB/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-0/.ATTRIBUTES/OBJECT_CONFIG_JSONB'layer-10/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-3/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-6/.ATTRIBUTES/OBJECT_CONFIG_JSONB&layer-9/.ATTRIBUTES/OBJECT_CONFIG_JSONB3layer_with_weights-0/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-1/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-2/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-3/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-4/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-5/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-6/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-7/.ATTRIBUTES/OBJECT_CONFIG_JSONB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
ł
!save_1/RestoreV2/shape_and_slicesConst"/device:CPU:0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
š
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices"/device:CPU:0*,
dtypes"
 2*
_output_shapesz
x::::::::::::::::::::::::::::::

save_1/NoOpNoOp

save_1/NoOp_1NoOp

save_1/NoOp_2NoOp

save_1/NoOp_3NoOp

save_1/NoOp_4NoOp

save_1/NoOp_5NoOp

save_1/NoOp_6NoOp
R
save_1/IdentityIdentitysave_1/RestoreV2:7*
_output_shapes
:*
T0
V
save_1/AssignVariableOpAssignVariableOpconv2d/biassave_1/Identity*
dtype0
T
save_1/Identity_1Identitysave_1/RestoreV2:8*
_output_shapes
:*
T0
\
save_1/AssignVariableOp_1AssignVariableOpconv2d/kernelsave_1/Identity_1*
dtype0

save_1/NoOp_7NoOp
U
save_1/Identity_2Identitysave_1/RestoreV2:10*
_output_shapes
:*
T0
\
save_1/AssignVariableOp_2AssignVariableOpconv2d_1/biassave_1/Identity_2*
dtype0
U
save_1/Identity_3Identitysave_1/RestoreV2:11*
T0*
_output_shapes
:
^
save_1/AssignVariableOp_3AssignVariableOpconv2d_1/kernelsave_1/Identity_3*
dtype0

save_1/NoOp_8NoOp
U
save_1/Identity_4Identitysave_1/RestoreV2:13*
_output_shapes
:*
T0
\
save_1/AssignVariableOp_4AssignVariableOpconv2d_2/biassave_1/Identity_4*
dtype0
U
save_1/Identity_5Identitysave_1/RestoreV2:14*
_output_shapes
:*
T0
^
save_1/AssignVariableOp_5AssignVariableOpconv2d_2/kernelsave_1/Identity_5*
dtype0

save_1/NoOp_9NoOp
U
save_1/Identity_6Identitysave_1/RestoreV2:16*
T0*
_output_shapes
:
\
save_1/AssignVariableOp_6AssignVariableOpconv2d_3/biassave_1/Identity_6*
dtype0
U
save_1/Identity_7Identitysave_1/RestoreV2:17*
T0*
_output_shapes
:
^
save_1/AssignVariableOp_7AssignVariableOpconv2d_3/kernelsave_1/Identity_7*
dtype0

save_1/NoOp_10NoOp
U
save_1/Identity_8Identitysave_1/RestoreV2:19*
T0*
_output_shapes
:
\
save_1/AssignVariableOp_8AssignVariableOpconv2d_4/biassave_1/Identity_8*
dtype0
U
save_1/Identity_9Identitysave_1/RestoreV2:20*
T0*
_output_shapes
:
^
save_1/AssignVariableOp_9AssignVariableOpconv2d_4/kernelsave_1/Identity_9*
dtype0

save_1/NoOp_11NoOp
V
save_1/Identity_10Identitysave_1/RestoreV2:22*
T0*
_output_shapes
:
^
save_1/AssignVariableOp_10AssignVariableOpconv2d_5/biassave_1/Identity_10*
dtype0
V
save_1/Identity_11Identitysave_1/RestoreV2:23*
T0*
_output_shapes
:
`
save_1/AssignVariableOp_11AssignVariableOpconv2d_5/kernelsave_1/Identity_11*
dtype0

save_1/NoOp_12NoOp
V
save_1/Identity_12Identitysave_1/RestoreV2:25*
T0*
_output_shapes
:
[
save_1/AssignVariableOp_12AssignVariableOp
dense/biassave_1/Identity_12*
dtype0
V
save_1/Identity_13Identitysave_1/RestoreV2:26*
T0*
_output_shapes
:
]
save_1/AssignVariableOp_13AssignVariableOpdense/kernelsave_1/Identity_13*
dtype0

save_1/NoOp_13NoOp
V
save_1/Identity_14Identitysave_1/RestoreV2:28*
_output_shapes
:*
T0
]
save_1/AssignVariableOp_14AssignVariableOpdense_1/biassave_1/Identity_14*
dtype0
V
save_1/Identity_15Identitysave_1/RestoreV2:29*
T0*
_output_shapes
:
_
save_1/AssignVariableOp_15AssignVariableOpdense_1/kernelsave_1/Identity_15*
dtype0
Ŕ
save_1/restore_allNoOp^save_1/AssignVariableOp^save_1/AssignVariableOp_1^save_1/AssignVariableOp_10^save_1/AssignVariableOp_11^save_1/AssignVariableOp_12^save_1/AssignVariableOp_13^save_1/AssignVariableOp_14^save_1/AssignVariableOp_15^save_1/AssignVariableOp_2^save_1/AssignVariableOp_3^save_1/AssignVariableOp_4^save_1/AssignVariableOp_5^save_1/AssignVariableOp_6^save_1/AssignVariableOp_7^save_1/AssignVariableOp_8^save_1/AssignVariableOp_9^save_1/NoOp^save_1/NoOp_1^save_1/NoOp_10^save_1/NoOp_11^save_1/NoOp_12^save_1/NoOp_13^save_1/NoOp_2^save_1/NoOp_3^save_1/NoOp_4^save_1/NoOp_5^save_1/NoOp_6^save_1/NoOp_7^save_1/NoOp_8^save_1/NoOp_9

init_1NoOp"J
save_1/Const:0save_1/control_dependency:0save_1/restore_all 5 @F8"!
saved_model_main_op


init_1"ż
	variablesąŽ
|
conv2d/kernel:0conv2d/kernel/Assign#conv2d/kernel/Read/ReadVariableOp:0(2*conv2d/kernel/Initializer/random_uniform:08
k
conv2d/bias:0conv2d/bias/Assign!conv2d/bias/Read/ReadVariableOp:0(2conv2d/bias/Initializer/zeros:08

conv2d_1/kernel:0conv2d_1/kernel/Assign%conv2d_1/kernel/Read/ReadVariableOp:0(2,conv2d_1/kernel/Initializer/random_uniform:08
s
conv2d_1/bias:0conv2d_1/bias/Assign#conv2d_1/bias/Read/ReadVariableOp:0(2!conv2d_1/bias/Initializer/zeros:08

conv2d_2/kernel:0conv2d_2/kernel/Assign%conv2d_2/kernel/Read/ReadVariableOp:0(2,conv2d_2/kernel/Initializer/random_uniform:08
s
conv2d_2/bias:0conv2d_2/bias/Assign#conv2d_2/bias/Read/ReadVariableOp:0(2!conv2d_2/bias/Initializer/zeros:08

conv2d_3/kernel:0conv2d_3/kernel/Assign%conv2d_3/kernel/Read/ReadVariableOp:0(2,conv2d_3/kernel/Initializer/random_uniform:08
s
conv2d_3/bias:0conv2d_3/bias/Assign#conv2d_3/bias/Read/ReadVariableOp:0(2!conv2d_3/bias/Initializer/zeros:08

conv2d_4/kernel:0conv2d_4/kernel/Assign%conv2d_4/kernel/Read/ReadVariableOp:0(2,conv2d_4/kernel/Initializer/random_uniform:08
s
conv2d_4/bias:0conv2d_4/bias/Assign#conv2d_4/bias/Read/ReadVariableOp:0(2!conv2d_4/bias/Initializer/zeros:08

conv2d_5/kernel:0conv2d_5/kernel/Assign%conv2d_5/kernel/Read/ReadVariableOp:0(2,conv2d_5/kernel/Initializer/random_uniform:08
s
conv2d_5/bias:0conv2d_5/bias/Assign#conv2d_5/bias/Read/ReadVariableOp:0(2!conv2d_5/bias/Initializer/zeros:08
x
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(2)dense/kernel/Initializer/random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/zeros:08

dense_1/kernel:0dense_1/kernel/Assign$dense_1/kernel/Read/ReadVariableOp:0(2+dense_1/kernel/Initializer/random_uniform:08
o
dense_1/bias:0dense_1/bias/Assign"dense_1/bias/Read/ReadVariableOp:0(2 dense_1/bias/Initializer/zeros:08"É
trainable_variablesąŽ
|
conv2d/kernel:0conv2d/kernel/Assign#conv2d/kernel/Read/ReadVariableOp:0(2*conv2d/kernel/Initializer/random_uniform:08
k
conv2d/bias:0conv2d/bias/Assign!conv2d/bias/Read/ReadVariableOp:0(2conv2d/bias/Initializer/zeros:08

conv2d_1/kernel:0conv2d_1/kernel/Assign%conv2d_1/kernel/Read/ReadVariableOp:0(2,conv2d_1/kernel/Initializer/random_uniform:08
s
conv2d_1/bias:0conv2d_1/bias/Assign#conv2d_1/bias/Read/ReadVariableOp:0(2!conv2d_1/bias/Initializer/zeros:08

conv2d_2/kernel:0conv2d_2/kernel/Assign%conv2d_2/kernel/Read/ReadVariableOp:0(2,conv2d_2/kernel/Initializer/random_uniform:08
s
conv2d_2/bias:0conv2d_2/bias/Assign#conv2d_2/bias/Read/ReadVariableOp:0(2!conv2d_2/bias/Initializer/zeros:08

conv2d_3/kernel:0conv2d_3/kernel/Assign%conv2d_3/kernel/Read/ReadVariableOp:0(2,conv2d_3/kernel/Initializer/random_uniform:08
s
conv2d_3/bias:0conv2d_3/bias/Assign#conv2d_3/bias/Read/ReadVariableOp:0(2!conv2d_3/bias/Initializer/zeros:08

conv2d_4/kernel:0conv2d_4/kernel/Assign%conv2d_4/kernel/Read/ReadVariableOp:0(2,conv2d_4/kernel/Initializer/random_uniform:08
s
conv2d_4/bias:0conv2d_4/bias/Assign#conv2d_4/bias/Read/ReadVariableOp:0(2!conv2d_4/bias/Initializer/zeros:08

conv2d_5/kernel:0conv2d_5/kernel/Assign%conv2d_5/kernel/Read/ReadVariableOp:0(2,conv2d_5/kernel/Initializer/random_uniform:08
s
conv2d_5/bias:0conv2d_5/bias/Assign#conv2d_5/bias/Read/ReadVariableOp:0(2!conv2d_5/bias/Initializer/zeros:08
x
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(2)dense/kernel/Initializer/random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/zeros:08

dense_1/kernel:0dense_1/kernel/Assign$dense_1/kernel/Read/ReadVariableOp:0(2+dense_1/kernel/Initializer/random_uniform:08
o
dense_1/bias:0dense_1/bias/Assign"dense_1/bias/Read/ReadVariableOp:0(2 dense_1/bias/Initializer/zeros:08*¤
serving_default
=
conv2d_input-
conv2d_input:0˙˙˙˙˙˙˙˙˙@@3
dense_1(
dense_1/Softmax:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict