# SphereDiscriminator

My recreation of Ian Goodfellow's [Adversarial Spheres](https://arxiv.org/abs/1801.02774) paper.

sphere_discriminator.py trains a model to predict if a point is on a sphere of radius 1 or a sphere or radius 1.3, then uses gradient descent to try to find a point on a sphere of radius 1 that the model misclassifies.  In accordance with the paper's findings, in lower dimensions, it is extremely difficult to produce adversarial examples.  Conversely, in higher dimensions, adversarial examples are easy to produce.

Run with:

`python sphere_discriminator.py <Number of Dimensions>`