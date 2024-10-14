CXX = g++
CXXFLAGS = -std=c++17
INCLUDES = -Iinclude

SRCS = src/main.cpp \
       src/nn/nn.cpp \
       src/nn/nn_functional.cpp \
       src/nn/optim.cpp \
       src/tensor/tensor.cpp \
	   src/data/dataset.cpp \
	   src/data/dataloader.cpp \

TARGET = build/nn_project

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^

clean:
	rm -f $(TARGET)