# Genetic Painting

## Related Work
### Genetic Drawing

* In each stage, draw `strokeCount = 10` strokes
* This 10 strokes formed a DNA, encoding each stroke's position, size, rotaion, type...
* The population only contain 1 DNA, the only update operation is **mutation**
* In each generation, pick one stroke in the DNA and mutate it, if better, keep it
* After several generations, draw the DNA on canvas, start the next generation

### Genetic Algorithm to Draw Images

* Compare single-parent and population-based



## Basic Algorithm

* Initialize the population
* Begin loop
  * Evaluate the Fitness
  * Generate the offspring
  * Environmental selection
* End loop

### What's the individual(DNA)? 

The DNA is a list of genes where each gene encodes a polygon. The polygon could be a square, circle, rectangle, ellipse, triangle, or N-vertex polygon. In addition each gene encodes the color, location (including z-index), transparency, and size of each polygon.

## Reference

* [Genetic Algorithm to Draw Images](https://kennycason.com/posts/2016-06-01-genetic-algorithm-draw-images.html)
* [Geometrize](https://www.geometrize.co.uk/)
* [Genetic Art Algorithm](https://blog.4dcu.be/programming/2020/01/12/Genetic-Art-Algorithm.html)
* [Parallel Genetic Algorithm](https://medium.com/swlh/parallel-genetic-algorithm-3d3314c8373c#:~:text=%20Parallel%20Genetic%20Algorithm%20%201%20Concept%20of,simply%20instruct%20multiple%20processors%20to%20create...%20More%20)
* [Genetic Drawing](https://github.com/anopara/genetic-drawing)
* [Genetic Programming: Evolution of Mona Lisa](https://rogerjohansson.blog/2008/12/07/genetic-programming-evolution-of-mona-lisa/)
* [Primitive for macOS](https://primitive.lol/)