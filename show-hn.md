Show HN: Jixp, a lisp dsl for describing Jax neural nets
This is a side project I've been working on while learning Jax.
I noticed that a bunch of the neural net math looked like it would work well in a functional syntax because most of the data flows through layers in a "functional" manner.
I've been using this while learning jax to toy around with different shapes/styles of models. I used this to train a 4m parameter model on my Obsidian vault to see if I could use it for recall, it worked relatively poorly.
The Jixp compiler reads the (racket style) DSL and outputs python which is then consumed via your normal python training stack. The advantage here is you get a pretty simple interface to describe your model and the implementation details are largely hidden in the generated python.
I definitely do not recommend using this for anything besides learning. Eventually I can see some sort of complier for neural nets which is then consumed by downstream users, but this would need a ton of work before it got there.

