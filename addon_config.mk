# All variables and this file are optional, if they are not present the PG and the
# makefiles will try to parse the correct values from the file system.
#
# Variables that specify exclusions can use % as a wildcard to specify that anything in
# that position will match. A partial path can also be specified to, for example, exclude
# a whole folder from the parsed paths from the file system
#
# Variables can be specified using = or +=
# = will clear the contents of that variable both specified from the file or the ones parsed
# from the file system
# += will add the values to the previous ones in the file or the ones parsed from the file 
# system
# 
# The PG can be used to detect errors in this file, just create a new project with this addon 
# and the PG will write to the console the kind of error and in which line it is

meta:
	ADDON_NAME = ofxDarknet
	ADDON_DESCRIPTION = Addon for darknet neural network framework
	ADDON_AUTHOR = Marcel Schwittlick
	ADDON_TAGS = "machine learning" "deep learning" "neural networks"
	ADDON_URL = https://github.com/mrzl/ofxDarknet

common:
	# dependencies with other addons, a list of them separated by spaces 
	# or use += in several lines
	ADDON_DEPENDENCIES = ofxOpenCv
	
	# include search paths, this will be usually parsed from the file system
	# but if the addon or addon libraries need special search paths they can be
	# specified here separated by spaces or one per line using +=
	ADDON_INCLUDES = libs
	ADDON_INCLUDES += src
	ADDON_INCLUDES += /usr/local/cuda/include
	
	# any special flag that should be passed to the compiler when using this
	# addon
	# ADDON_CFLAGS =
	
	# any special flag that should be passed to the linker when using this
	# addon, also used for system libraries with -lname
	# ADDON_LDFLAGS =
	
	# linux only, any library that should be included in the project using
	# pkg-config
	# ADDON_PKG_CONFIG_LIBRARIES =
	
	# osx/iOS only, any framework that should be included in the project
	# ADDON_FRAMEWORKS =
	
	# source files, these will be usually parsed from the file system looking
	# in the src folders in libs and the root of the addon. if your addon needs
	# to include files in different places or a different set of files per platform
	# they can be specified here
	ADDON_SOURCES = src/ofxDarknet.cpp
	ADDON_SOURCES += src/ofxDarknet.h
	ADDON_SOURCES += libs/darknet/include/activation_layer.h
	ADDON_SOURCES += libs/darknet/include/activations.h
	ADDON_SOURCES += libs/darknet/include/avgpool_layer.h
	ADDON_SOURCES += libs/darknet/include/batchnorm_layer.h
	ADDON_SOURCES += libs/darknet/include/blas.h
	ADDON_SOURCES += libs/darknet/include/box.h
	ADDON_SOURCES += libs/darknet/include/classifier.h
	ADDON_SOURCES += libs/darknet/include/col2im.h
	ADDON_SOURCES += libs/darknet/include/connected_layer.h
	ADDON_SOURCES += libs/darknet/include/convolutional_layer.h
	ADDON_SOURCES += libs/darknet/include/cost_layer.h
	ADDON_SOURCES += libs/darknet/include/crnn_layer.h
	ADDON_SOURCES += libs/darknet/include/crop_layer.h
	ADDON_SOURCES += libs/darknet/include/cuda.h
	ADDON_SOURCES += libs/darknet/include/data.h
	ADDON_SOURCES += libs/darknet/include/deconvolutional_layer.h
	ADDON_SOURCES += libs/darknet/include/demo.h
	ADDON_SOURCES += libs/darknet/include/detection_layer.h
	ADDON_SOURCES += libs/darknet/include/dropout_layer.h
	ADDON_SOURCES += libs/darknet/include/gemm.h
	ADDON_SOURCES += libs/darknet/include/getopt.h
	ADDON_SOURCES += libs/darknet/include/gettimeofday.h
	ADDON_SOURCES += libs/darknet/include/gru_layer.h
	ADDON_SOURCES += libs/darknet/include/im2col.h
	ADDON_SOURCES += libs/darknet/include/image.h
	ADDON_SOURCES += libs/darknet/include/layer.h
	ADDON_SOURCES += libs/darknet/include/list.h
	ADDON_SOURCES += libs/darknet/include/local_layer.h
	ADDON_SOURCES += libs/darknet/include/matrix.h
	ADDON_SOURCES += libs/darknet/include/maxpool_layer.h
	ADDON_SOURCES += libs/darknet/include/network.h
	ADDON_SOURCES += libs/darknet/include/nightmare.h
	ADDON_SOURCES += libs/darknet/include/normalization_layer.h
	ADDON_SOURCES += libs/darknet/include/option_list.h
	ADDON_SOURCES += libs/darknet/include/parser.h
	ADDON_SOURCES += libs/darknet/include/region_layer.h
	ADDON_SOURCES += libs/darknet/include/reorg_layer.h
	ADDON_SOURCES += libs/darknet/include/rnn_layer.h
	ADDON_SOURCES += libs/darknet/include/rnn_vid.h
	ADDON_SOURCES += libs/darknet/include/rnn.h
	ADDON_SOURCES += libs/darknet/include/route_layer.h
	ADDON_SOURCES += libs/darknet/include/shortcut_layer.h
	ADDON_SOURCES += libs/darknet/include/softmax_layer.h
	ADDON_SOURCES += libs/darknet/include/stb_image_write.h
	ADDON_SOURCES += libs/darknet/include/stb_image.h
	ADDON_SOURCES += libs/darknet/include/tree.h
	ADDON_SOURCES += libs/darknet/include/unistd.h
	ADDON_SOURCES += libs/darknet/include/utils.h

	# some addons need resources to be copied to the bin/data folder of the project
	# specify here any files that need to be copied, you can use wildcards like * and ?
	# ADDON_DATA = 
	
	# when parsing the file system looking for libraries exclude this for all or
	# a specific platform
	# ADDON_LIBS_EXCLUDE =

	ADDON_LIBS_INCLUDE = /usr/helloworld
	
msys2:

