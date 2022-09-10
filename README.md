# Two_Link_System
Designing a controller under the assumption that the structure is
completely unknown for 𝜏!(∅, ∅̇ +\
From the dynamics,\
𝑀(∅)∅̈ + 𝐶(∅, ∅̇ + + 𝐺(∅) + 𝜏!(∅, ∅̇ + = 𝜏\
𝑀(∅)∅̈ + 𝐶(∅, ∅̇ + + 𝐺(∅) accounts to the structured uncertainties and
𝜏!(∅, ∅̇ + represents the unstructured uncertainties. And as a result, we
can design a linear in the parameter design for the structured
uncertainties. And the unstructured uncertainties can be estimated using
deep neural network.

For deep neural network 𝜏!(∅, ∅̇ + can be estimated as\
𝜏!(∅, ∅̇ + = 𝑊\*𝜎(Φ(∅, ∅̇ + + 𝜀(∅, ∅̇ +\
Where 𝑊𝜖 ℝ+×\#is the output weights, 𝜎(. )𝜖 ℝ+ is the activation
function, Φis the inner and the reconstruction error is 𝜀(∅, ∅̇ + 𝜖 ℝ\#.

Φ\],- = 𝜎,(𝑉\_,-\*𝜙\_,.\" - + 𝑏\_,-)\
Error Dynamics-\
𝑒 = ∅! − ∅\
𝑒̇ = ∅! − ∅\
𝑒̈ = ∅! − ∅\
Reference tracking error\
𝑟 = 𝑒̇ + 𝛼𝑒\
𝑟̇ = 𝑒̈ + 𝛼𝑒̇\
𝑟̇ = ∅! − ∅̈ + 𝛼𝑒̇\
Multiplying both sides by 𝑀(∅) we get,\
𝑀(∅)𝑟̇ = 𝑀(∅)∅! − 𝑀(∅)∅̈ + 𝑀(∅)𝛼𝑒̇\
From the dynamics we know,\
𝑀(∅)∅̈ = −𝐶(∅, ∅̇ + − 𝐺(∅) − 𝜏!(∅, ∅̇ + + 𝜏\
On substitution we get,\
𝑀(∅)𝑟̇ = 𝑀(∅)∅! − 𝑀(∅)∅̈ + 𝑀(∅)𝛼𝑒̇\
𝑀(∅)𝑟 = 𝑀(∅)∅! − (−𝐶(∅, ∅̇ + − 𝐺(∅) − 𝜏!(∅, ∅̇ + + 𝜏) + 𝑀(∅)𝛼𝑒̇ \"\
\#𝑀̇ (∅, ∅̇ +𝑟Add and subtract by\
𝑀(∅)𝑟̇ = 𝑀(∅)∅! + 𝐶(∅, ∅̇ + + 𝐺(∅) + 𝜏!(∅, ∅̇ + − 𝜏 + 𝑀(∅)𝛼𝑒̇ ± 1~~2~~ 𝑀̇ (∅,
∅̇ +𝑟 𝑀(∅)𝑟̇ = 𝑀(∅)(∅! + 𝛼𝑒̇) + 𝐶(∅, ∅̇ + + 𝐺(∅) + 𝜏!(∅, ∅̇ + − 𝜏 ± 1~~2~~ 𝑀̇
(∅, ∅̇ +𝑟

\"\
We can estimate 𝑀(∅) (∅! + 𝛼𝑒̇) + 𝐶(∅, ∅̇ + + 𝐺(∅) +\#𝑀̇ (∅, ∅̇ +𝑟 = 𝑌𝜃
since it is linear in the unknown parameters and we were able to develop
the unknown parameters 𝜃 as follows, \# + 𝑚\#𝑙\"\# + 𝑚\#𝑙\#\#⎡𝜃\" ⎢ ⎢ ⎢
𝜃\# 𝜃0 𝜃( ⎤ ⎥ ⎥ ⎥ = ⎡𝑚\"𝑙\" ⎢ ⎢ ⎢ ⎢ (𝑚\" + 𝑚\#)𝑙\" 𝑚\#𝑙\"𝑙\# 𝑚\#𝑙\" \# ⎤
⎥ ⎥ ⎥ ⎥\
⎣𝜃1⎦ ⎣ 𝑚\#𝑙\# ⎦

And 𝑌 = 𝑌2(∅, ∅! + 𝛼𝑒̇+ + 𝑌3(∅, ∅̇ + + 𝑌4(∅) +

> \"

The unstructured state-dependent disturbance 𝜏!(∅, ∅̇ + will be estimated
using a two-layer neural network.

𝜏!(∅, ∅̇ + = 𝑊\*𝜎(Φ(𝜁) + 𝜀(∅, ∅̇ +\
Now,\
𝑀(∅)𝑟̇ = 𝑌𝜃 + 𝑊\*𝜎(Φ(𝜁)) + 𝜀(∅, ∅̇ + − 𝜏 − 1~~2~~ 𝑀̇ (∅, ∅̇ +𝑟\
Approximation for\
~~𝑑Φ~~ Φo + 𝜀6(Φo \#)𝜎(Φ) = 𝜎(Φ\]+ + 𝑑𝜎\
Where,\
Φo = Φ − Φ\] = 𝑉\*𝜁 − 𝑉\_ \*𝜁 = 𝑉p𝜁\
𝜎 = 𝜎 + 𝑑𝜎~~𝑑Φ~~ Φo + 𝜀6\
Now on substitution we get,\
~~2~~ 𝑀̇ (∅, ∅̇ +𝑟𝑀(∅)𝑟̇ = 𝑌𝜃 + 𝑊\*(𝜎(Φ\]+ + 𝑑𝜎~~𝑑Φ~~ Φo + 𝜀6(Φo \#)) +
𝜀(∅, ∅̇ + − 𝜏 − 1\
Add and subtract 𝑊\] \*𝜎q7Φo

𝑀(∅)𝑟̇ = 𝑌𝜃 + 𝑊\*r𝜎(Φ\]+ + 𝑑𝜎~~𝑑Φ~~ Φo + 𝜀6(Φo \#+s + 𝜀(∅, ∅̇ + − 𝜏 −
1~~2~~ 𝑀̇ (∅, ∅̇ +𝑟 ± 𝑊\] \*𝜎q7Φo

𝑀(∅)𝑟̇ = 𝑌𝜃 + 𝑊\*𝜎(Φ\]+ − 𝜏 − 1~~2~~ 𝑀̇ (∅, ∅̇ +𝑟 + 𝑊\] \*𝜎q7Φo + 𝑊o
\*𝜎q7Φo + 𝑊\*𝜀6(Φo \#+ + 𝜀(∅, ∅̇ +Let\
𝑊o \*𝜎q7Φo + 𝑊\*𝜀6(Φo \#+ + 𝜀(∅, ∅̇ + = 𝛿\
𝑀(∅)𝑟̇ = 𝑌𝜃 + 𝑊\*𝜎(Φ\]+ − 𝜏 − 1~~2~~ 𝑀̇ (∅, ∅̇ +𝑟\
Let the stacked errors\
𝑒\
𝑟\
𝜉 = v 𝜃p x 𝜖 ℝ\#8\#818\#+81,\
𝑣𝑒𝑐(𝑊o )

Where,\
⎡𝑊\"\" y ⎤\
𝑣𝑒𝑐(𝑊o + = ⎢ ⎢ ⎢ 𝑊+\" 𝑊\"\# y y ⋮ ⎥ ⎥ ⎥\
⎢ ⋮ ⎥\
⎣𝑊+\# y ⎦\
Lyapunov Candidate,\
𝑉 = 1~~2~~ 𝑒\*𝑒 + 1~~2~~ 𝑟\*𝑀(∅)𝑟 + 1~~2~~ 𝜃p\*Γ9.\"𝜃p + 1~~2~~ 𝑡𝑟(𝑊o
\*Γ:.\"𝑊o ) 𝑉̇ = 𝑒\*𝑒 + 1~~2~~ 𝑟\*𝑀̇ (∅)𝑟 + 𝑟\*𝑀𝑟̇ + 𝜃p\*Γ9.\"𝜃p + 𝑡𝑟(𝑊o
\*Γ:.\"𝑊o +

We know,

> 𝑟 = 𝑒̇ = 𝑟 − 𝛼𝑒\
> 𝜃̇p = −𝜃̇\_\
> 𝑊̇o = −𝑊̇\]

And\
𝑀(∅)𝑟̇ = 𝑌𝜃 + 𝑊\*𝜎(Φ\]+ − 𝜏 − 1~~2~~ 𝑀̇ (∅, ∅̇ +𝑟 On substitution we get,\
𝑉̇ = 𝑒\*𝑒 + 1~~2~~ 𝑟\*𝑀̇ (∅)𝑟 + 𝑟\*𝑀𝑟̇ + 𝜃p\*Γ9.\"𝜃p + 𝑡𝑟 :𝑊o \*Γ:.\"𝑊̇o ?

𝑉̇ = 𝑒\*𝑒̇ + 1~~2~~ 𝑟\*𝑀̇ (∅)𝑟 + 𝑟\*(𝑌𝜃 + 𝑊\*𝜎(Φ\]+ − 𝜏 − 1~~2~~ 𝑀̇ (∅, ∅̇
+𝑟) + 𝜃p\*Γ9.\"𝜃̇p + 𝑡𝑟 :𝑊o \*Γ:.\"𝑊̇o ?

On simplification we get,\
.\"𝑊̇\] ?𝑉̇ = 𝑒\*(𝑟 − 𝛼𝑒) + 𝑟\*(𝑌𝜃 + 𝑊\*𝜎(Φ\]+ − 𝜏 + − 𝜃p\*Γ9.\"𝜃̇\_ − 𝑡𝑟
:𝑊o \*Γ:

Design the input 𝜏\
𝜏 = 𝑒 + 𝑌𝜃\_ + 𝑊\] \*𝜎q- + 𝛽;𝑠𝑔𝑛(𝑟) + 𝛽\<𝑟\
𝑉̇ = 𝑒\*(𝑟 − 𝛼𝑒) + 𝑟\*(𝑌𝜃 + 𝑊\*𝜎(Φ\]+ − (𝑒 + 𝑌𝜃\_ + 𝑊\] \*𝜎q- + 𝛽;𝑠𝑔𝑛(𝑟)
+ 𝛽\<𝑟)+ − 𝜃p\*Γ9.\"𝜃̇\_ − 𝑡𝑟 :𝑊o \*Γ:.\"𝑊̇\] ?

𝑉̇ = −(𝑒\*𝛼𝑒) + 𝑟\*𝑌𝜃p + 𝑟\*𝑊o \*𝜎q- − 𝑟\*𝛽;𝑠𝑔𝑛(𝑟) − 𝑟\*𝛽\<𝑟 −
𝜃p\*Γ9.\"𝜃̇\_ − 𝑡𝑟 :𝑊o \*Γ:.\"𝑊̇\] ?

Design for 𝜃̇\_, we want to get,

So,

Design for 𝑊̇\], we want to get,

> 𝑟\*𝑌𝜃p − 𝜃p\*Γ9.\"𝜃̇\_ = 0\
> 𝜃p\*Γ9.\"𝜃̇\_ = 𝑟\*𝑌𝜃p
>
> 𝜃̇\_ = proj(Γ9𝑌\*𝑟)
>
> 𝑟\*𝑊o \*σ‚ − 𝑡𝑟 :𝑊o \*Γ:.\"𝑊̇\] ? = 0

We know,\
𝑡𝑟(𝑏𝑎\*) = 𝑎\*𝑏\
𝑡𝑟(𝑊o \*σ‚𝑟\*+ = 𝑡𝑟(𝑊o \*Γ:.\"𝑊̇\] )So,\
𝑊̇\] = 𝑝𝑟𝑜𝑗(Γ:𝜎q-†𝑟\*)Yielding\
𝑉,= ≤ −(𝑒\*𝛼𝑒) − 𝑟\*𝛽\<𝑟 =%\>,!?

Using Barbalat's Lemma we can show the ‖𝑒‖, ‖𝑟‖ˆ⎯⎯⎯Š 0 which implies
Asymptotic tracking

Simulation results

> 1)The first 4 DNN choices were a good first choice for the
> architectures because they were easy to implement and these function
> restricted the bounds of the data to be in the range of -1 to 1,
> ensuring that the function does not become unstable.
>
> 2)After selecting different activation function for each layer, the
> results showed that linear ReLus with a different output layer give
> the most accurate approximation of the function, so I used multiple
> ReLu with a 2 layer of the Sigmoid at the end for the last design.
>
> 3)Norm of tracking error and filter tracking error From Two-layer
> Neural network
>
> ![](vertopal_84a8f5d296e74188ac300df24ca140fa/media/image1.png){width="5.277777777777778in"
> height="3.7083333333333335in"}
>
> ![](vertopal_84a8f5d296e74188ac300df24ca140fa/media/image2.png){width="5.277777777777778in"
> height="3.7222222222222223in"}
>
> From Gaussian activation functions
>
> ![](vertopal_84a8f5d296e74188ac300df24ca140fa/media/image3.png){width="5.583333333333333in"
> height="3.9722222222222223in"}
>
> For Tanh activation function
>
> ![](vertopal_84a8f5d296e74188ac300df24ca140fa/media/image4.png){width="5.583333333333333in"
> height="4.388888888888889in"}
>
> From Relu with Sigmoid output
>
> ![](vertopal_84a8f5d296e74188ac300df24ca140fa/media/image5.png){width="5.583333333333333in"
> height="4.402777777777778in"}

ReLu with Tanh output

> ![](vertopal_84a8f5d296e74188ac300df24ca140fa/media/image6.png){width="5.583333333333333in"
> height="4.388888888888889in"}

From more ReLu with 2 Sigmoid layers

> ![](vertopal_84a8f5d296e74188ac300df24ca140fa/media/image7.png){width="5.583333333333333in"
> height="4.388888888888889in"}

For the graphs it can be seen that the Relu with sigmoid output
performed the best with 1.5 RMS error compared to the other
architectures. But all the other design performed well in tracking the
objective.

\(4\) Norm of the function approximation

For Gaussian activation

> ![](vertopal_84a8f5d296e74188ac300df24ca140fa/media/image8.png){width="5.638888888888889in"
> height="4.388888888888889in"}

For Tanh activation

![](vertopal_84a8f5d296e74188ac300df24ca140fa/media/image9.png){width="0.18055555555555555in"
height="0.6388888888888888in"}

> ![](vertopal_84a8f5d296e74188ac300df24ca140fa/media/image10.png){width="5.430555555555555in"
> height="4.402777777777778in"}

For ReLu with Sigmoid output

> ![](vertopal_84a8f5d296e74188ac300df24ca140fa/media/image11.png){width="4.513888888888889in"
> height="3.5277777777777777in"}

For ReLu with Tanh output

> ![](vertopal_84a8f5d296e74188ac300df24ca140fa/media/image12.png){width="4.069444444444445in"
> height="3.1805555555555554in"}

Any design

> ![](vertopal_84a8f5d296e74188ac300df24ca140fa/media/image13.png){width="5.055555555555555in"
> height="3.3333333333333335in"}
>
> The last three design seem to perform the best.

\(5\) Input plots

\(a\)

![](vertopal_84a8f5d296e74188ac300df24ca140fa/media/image14.png){width="0.20833333333333334in"
height="0.4305555555555556in"}

> ![](vertopal_84a8f5d296e74188ac300df24ca140fa/media/image15.png){width="5.430555555555555in"
> height="4.388888888888889in"}

\(b\)

> ![](vertopal_84a8f5d296e74188ac300df24ca140fa/media/image16.png){width="5.652777777777778in"
> height="4.388888888888889in"}

\(c\)

![](vertopal_84a8f5d296e74188ac300df24ca140fa/media/image14.png){width="0.20833333333333334in"
height="0.4305555555555556in"}

> ![](vertopal_84a8f5d296e74188ac300df24ca140fa/media/image17.png){width="5.430555555555555in"
> height="4.388888888888889in"}

\(d\)

> ![](vertopal_84a8f5d296e74188ac300df24ca140fa/media/image18.png){width="5.736111111111111in"
> height="4.388888888888889in"}

+---+-----------------------------------------------------------------+
| € |                                                                 |
|   | -- ------------------------------------------------------------ |
|   | --------------------------------------------------------------- |
|   | --------------------------------------------------------------- |
|   | --------------------------------------------------------------- |
|   |      ![](vertopal_84a8f5d296e74188ac300df24ca140fa/media/image1 |
|   | 9.png){width="0.20833333333333334in" height="0.4305555555555556 |
|   | in"}![](vertopal_84a8f5d296e74188ac300df24ca140fa/media/image20 |
|   | .png){width="5.430555555555555in" height="4.402777777777778in"} |
|   |                                                                 |
|   | -- ------------------------------------------------------------ |
|   | --------------------------------------------------------------- |
|   | --------------------------------------------------------------- |
|   | --------------------------------------------------------------- |
|   |                                                                 |
|   | €                                                               |
+---+-----------------------------------------------------------------+
