
SRCS = Dubins/Dubins.cpp Dubins/DubinsIterator.cpp Dubins/dubins_iterator_c_api.cpp
OBJS = $(SRCS:.cpp=.o)
TARGET = libdubins.so

CXX = g++
CXXFLAGS = -fPIC -Wall -O2
LDFLAGS = -shared

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)