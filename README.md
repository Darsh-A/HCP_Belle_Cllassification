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


## Install

1. Setup and activate env

`python -m venv venv`

On Windows : `.\venv\Scripts\activate`


2. Install dependencies

`pip install -r requirements.txt`

### FastBDT
1. `git clone https://github.com/thomaskeck/FastBDT`

2. in a Linux environment:

`cmake .`

Add these lines to specific files:

include/FastBDT.h : 
`#include <cstdint>`
`#include <limits>`

src/FastBDT.cxx  right after `#include "FastBDT_IO.h"` : 
`#include <cstdint`

Run: `make`

Run: 
```
sudo apt-get update
sudo apt-get install rpm
```

Run: `sudo make install`

Run: `make package`

Run: `sudo python3 setup.py install`
OR
`pip install .`

Test: `python3 -c 'import PyFastBDT; print("PyFastBDT was installed successfully!")'`


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

Approx 2GB

You might need to install some dependencies if it fails: ` sudo apt install scons gfortran python3-dev`

8. Install the externals (dependencies)

`b2install-externals <external version> # Example b2install-externals 02-02-04`

Approx 6.5GB

You can check the external version by checking the error message in `b2setup release-<version>`

9. Run the software

`basf2 --info`

### Running Basf2

1. Start a WSL environment in the project directory (in whatever code editior like vs code)

2. Source and Setup the basf2 env

`source <tools_dir>/tools/b2setup`

Example:

`source /mnt/a/DSci/Projects/HEP_Event_Classf/tools/b2setup`

`b2setup release-09-00-04`

3. Run basf2 with a script
`basf2 <script.py> # Example basf2 myscript.py`
