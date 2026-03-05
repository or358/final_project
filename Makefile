# Makefile for the symnmf C program

CC = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors
LDFLAGS = -lm

symnmf: symnmf.o
	$(CC) -o symnmf symnmf.o $(LDFLAGS)

symnmf.o: symnmf.c symnmf.h
	$(CC) -c $(CFLAGS) symnmf.c

clean:
	rm -f *.o symnmf