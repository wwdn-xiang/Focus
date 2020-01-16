TF_INC = $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB = $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
all: clean build

build: woperation.so

woperation.so: wtoolkit.cc
	echo $^
	g++ -std=c++14 -Wno-ignored-attributes -Wno-unused-result -shared -g -o $@ $^ -I $(TF_INC) -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -L${TF_LIB} -lopencv_core -ltensorflow_framework -lopencv_imgproc `pkg-config --cflags --libs opencv`

clean:
	rm -f *.o *.so *.pyc *.npy
