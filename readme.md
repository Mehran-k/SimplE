Summary
=======

This software can be used to reproduce the results in our "SimplE Embedding for Link Prediction in Knowledge Graphs" paper. It can be also used to learn `SimplE` models for other datasets. The software can be also used as a framework to implement new tensor factorization models (implementations for `TransE` and `ComplEx` are included as two examples).

## Dependencies

* `Python` version 2.7 or higher
* `Numpy` version 1.13.1 or higher
* `Tensorflow` version 1.1.0 or higher

## Usage

To run a model `M` on a dataset `D`, do the following steps:
* `cd` to the directory where `main.py` is
* Run `python main.py -m M -d D`

Examples (commands start after $):

    $ python main.py -m SimplE_ignr -d wn18
    $ python main.py -m SimplE_avg -d fb15k
    $ python main.py -m ComplEx -d wn18

Running a model `M` on a dataset `D` will save the embeddings in a folder with the following address:

    $ <Current Directory>/M_weights/D/

As an example, running the `SimplE_ignr` model on `wn18` will save the embeddings in the following folder:

    $ <Current Directory>/SimplE_ignr_weights/wn18/

## Learned Embeddings for SimplE

The best embeddings learned for `SimplE_ignr` and `SimplE_avg` on `wn18` and `fb15k` can be downloaded from [this link](https://drive.google.com/file/d/1fSxdFbSIcS4w4mAHUhKewjmXCcbOGqM7/view?usp=sharing) and [this link](https://drive.google.com/file/d/1hpDS34BxNfbr6xGeut5q5nvx8fW98qCe/view?usp=sharing) respectively.

To use these embeddings, place them in the same folder as `main.py`, load the embeddings and use them.

## Publication

Refer to the following publication for details of the models and experiments.

- [Seyed Mehran Kazemi](https://mehran-k.github.io/) and [David Poole](http://www.cs.ubc.ca/~poole)

  [SimplE Embedding for Link Prediction in Knowledge Graphs](https://papers.nips.cc/paper/7682-simple-embedding-for-link-prediction-in-knowledge-graphs)
  
  [Representing and learning relations and properties under uncertainty](https://open.library.ubc.ca/collections/ubctheses/24/items/1.0375812)


## Cite SimplE

If you use this package for published work, please cite one (or both) of the following:

    @inproceedigs{kazemi2018simple,
      title={SimplE Embedding for Link Prediction in Knowledge Graphs},
      author={Kazemi, Seyed Mehran and Poole, David},
      booktitle={Advances in Neural Information Processing Systems},
      year={2018}
    }
    
    @phdthesis{Kazemi_2018, 
      series={Electronic Theses and Dissertations (ETDs) 2008+}, 
      title={Representing and learning relations and properties under uncertainty}, 
      url={https://open.library.ubc.ca/collections/ubctheses/24/items/1.0375812}, 
      DOI={http://dx.doi.org/10.14288/1.0375812}, 
      school={University of British Columbia}, 
      author={Kazemi, Seyed Mehran}, 
      year={2018}, 
      collection={Electronic Theses and Dissertations (ETDs) 2008+}
    }

Contact
=======

Seyed Mehran Kazemi

Computer Science Department

The University of British Columbia

201-2366 Main Mall, Vancouver, BC, Canada (V6T 1Z4)  

<http://www.cs.ubc.ca/~smkazemi/>  

<smkazemi@cs.ubc.ca>


License
=======

Licensed under the GNU General Public License Version 3.0.
<https://www.gnu.org/licenses/gpl-3.0.en.html>


Copyright (C) 2018  Seyed Mehran Kazemi

  





