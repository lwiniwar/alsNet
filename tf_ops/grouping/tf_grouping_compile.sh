#/bin/bash
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
/opt/sw/x86_64/glibc-2.17/ivybridge-ep/cuda/9.0.103/bin/nvcc tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I $TF_INC -I /opt/sw/x86_64/glibc-2.17/ivybridge-ep/cuda/9.0.103/include -lcudart -L /opt/sw/x86_64/glibc-2.17/ivybridge-ep/cuda/9.0.103/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -I $TF_INC/external/nsync/public -L $TF_LIB -ltensorflow_framework
