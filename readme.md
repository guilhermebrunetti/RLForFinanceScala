# Reinforcement Learning for Finance in Scala

This is my attempt to implement in Scala the code examples from book [Foundations of Reinforcement Learning with Applications in Finance][recent-pdf] by Ashwin Rao and Tikhon Jelvis.

[recent-pdf]: https://stanford.edu/~ashlearn/RLForFinanceBook/book.pdf

I really enjoyed reading this book, in particular the code examples, written in Python and available in this [link][rl-github-link]. The authors implemented many important algorithms from scratch using only Numpy and Scipy and used a Functional Programming style that I really appreciate. 

[rl-github-link]: https://github.com/TikhonJelvis/RL-book/tree/master/rl

Given my (limited) knowledge of Scala, I envisioned reimplementing the RL algorithms in Scala using their book as a guide.

I tried to replicate the package structure as much as possible. I renamed some parts of the Python code to cope with Scala Camelcase naming standard.

I used package Breeze for Numerical routines and Linear Algebra. In many situations, a Numpy formula has a clear equivalent in Breeze, but in other situations it is not clear. In particular, Numpy usually represents Vectors and Matrices with the same class (numpy.array), while Breeze has different classes for Vectors and Matrices.

The Static Type and Inheritance rules in Scala made it very hard for me to implement the generic Vector Spaces used in the Function Approximation class. I tried my best to find a solution that could be reused among Child classes, but I am not satisfied with the result. I think it could be written more elegantly but I don´t know how.

Please leave comments and suggestions, and read the original material for reference.