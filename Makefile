CC=g++
CFLAGS=-I.

# Source and output files
SRC = nn.cpp
OBJ = $(SRC:.cpp=.o)
TARGET = nn

# Default target
all: $(TARGET)

# Compile object files, using pattern rules to handle dependencies
%.o: %.cpp $(DEPS)
	$(CC) -c $< -o $@ $(CFLAGS)

# Link object files to create the final executable
$(TARGET): $(OBJ)
	$(CC) -o $(TARGET) $(OBJ)

# Clean up generated files
clean:
	rm -f $(OBJ) $(TARGET)

# Add a .PHONY target to avoid conflicts with files named "clean"
.PHONY: all clean