# Event Classification for HEP



This repo holds the code for the Data Science (IDC409 (Intro to DS and ML)) project of Group 23 



-----------



## Info





### Event Type Flags:





$e^+ e^- \\to \\Upsilon (4S) \\to B^+ B^-$ = 0



$e^+ e^- \\to \\Upsilon(4S) \\to B^0 \\bar{B}^0$ = 1



$e^+ e^- \\to c \\bar{c}$ = 2



$e^+ e^- \\to u \\bar{u}$ = 3



$e^+ e^- \\to d \\bar{d}$ = 4



$e^+ e^- \\to s \\bar{s}$ = 5





For binary add (0 and 1) to one class and (2,3,4, and 5) to other class



## Setting up Basf2

1. Run WSL in the Project directory

`wsl <path to project directory>`

2. Clone the Tools Repo

`git clone https://github.com/belle2/tools.git`

3. Add the path to your .bashrc file

`export PATH=/mnt/a/DSci/Projects/HEP_Event_Classf/tools:$PATH`

4. Source .bashrc

`source ~/.bashrc`

5. Give permission to the dir

`chmod +x /mnt/a/DSci/Projects/HEP_Event_Classf/tools/b2*`

6. Source basf2

`source b2setup`

7. Check releases

`b2install-release`

If you get a SHH error, run:

`git config --global url."https://github.com/".insteadOf git@github.com:`

Then run the release command again

Choose a release and install it by 

`b2install-release <release version> # Example b2install-release 09-08-04`

