# HOTS
HOTS: Hierarchy Of Time-Surfaces

HOTS is an approach developed by Xavier Lagorce, Garrick Orchard, Francesco Galluppi, Bertram E Shi, and Ryad B Benosman.
[(paper)](https://www.neuromorphic-vision.com/public/publications/1/publication.pdf).

This implementation is not very efficient as it is in Python and does not make use of acceleration frameworks such as Numba or Cython, however, this implementation offers code that is very easy to read and some plots that help to better understand the approach.

---

Event-based cameras are a new typology of vision sensors which offer some advantages w.r.t. standard camera sensors, mainly a high dynamic range and temporal resolution, low power consumption and low transfer rate as when there is no motion the sensor does not output any data.

<p align="left">
    <img src="images/ebc.png?raw=true" width="500" alt="Event Based Camera Advantages"/> </br>
</p>

The problem with this new type of cameras is that all state of the art algorithms were developed using standard images. Therefore, since their invention, many researchers have tried to develop algorithms for these kind of sensors as their properties make them a perfect fit for autonomous vehicles, robot navigation or other UAV applications were power supply is limited and real-time computations are needed.

This new approach for Pattern Recognition combines ideas of neighborhoods and temporal contexts. Below you can find an overview of the approach:

<p align="left">
    <img src="images/hots.jpg?raw=true" width="600" alt="HOTS overview"/> </br>
</p>
