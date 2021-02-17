'Dynamic General Equilibrium Modeling: Computational Methods and Applications'
(2nd edition, 2009)

By Burkhard Heer and and Alfred Maußner --- Python Code for Chapter 8



The files in Python_Chapter8_heer_maussner_dsge.rar are:
========================================================

readme_ch8_python.txt: this file

RCh8_part.py:	main program file, described in Chapter 8


Instructions:
=============

In order to run the Python programs, you simply need to store the file in a directory and run it. You also 
need to install libraries numpy, scipy, and linearmodels from Sargent and Starchuski:

"!pip install linearmodels"

Run time: approximately 5 hours : 30 minutes

Different from Algorithm 8.2.1, I use Algorithm 7.2.3 instead of Algorithm 7.2.2 in Step 5. The results are identical. 
However, I do not have to compute the inverse of the policy function and save computational time.

If this is your first PYTHON application from our book, we recommend to read the tutorial to the program RCh7_denf.g first. 
The two programs apply similar numerical methods. The program RCh7_denf.py is described in much more detail in our tutorial:

https://assets.uni-augsburg.de/media/filer_public/b0/4d/b04d79b7-2609-40ac-870b-126aada8e3f4/script_dge_python_11jan2021.html

In order to run the program, you should have installed Python and access to the library numpy.

A good introduction into Python and a description of the installation of Python can be found on
the website of Thomas Sargent and John Starchuski

https://python-programming.quantecon.org/intro.html

In case of questions, please send me an email: Burkhard.Heer@wiwi.uni-augsburg.de

last update: January 14, 2021