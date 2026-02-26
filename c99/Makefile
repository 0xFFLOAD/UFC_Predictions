CC = cc
CFLAGS = -std=c99 -Wall -Wextra -pedantic
TARGET = hello

all: $(TARGET)

$(TARGET): main.c
	$(CC) $(CFLAGS) -o $(TARGET) main.c

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
