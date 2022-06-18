# ⚙ MacOS Apple Silicon | Environment Setup
---

## STEP 1

**Download and install the Anaconda distribution.**
**Anaconda:**
1. Go to the [Anaconda product distribution page](https://www.anaconda.com/products/distribution)
2. Select at the MacOS 64-Bit(M1) Graphical Installer or Command Line Installer (316 and 305 MB) respectively.
3. Copy the URL and `wget <URL>`
    - At the time of setup, I got this URL:
      (https://repo.anaconda.com/archive/Anaconda3-2022.05-MacOSX-arm64.sh)
      
   ````zsh
   wget https://repo.anaconda.com/archive/Anaconda3-2022.05-MacOSX-arm64.sh
   ````
   
4. Find the filename of the installer or type `ls`
5. Run the installer with `bash <filename>` or `zsh <filename>`
6. Follow the on-screen instructions.
7. Log out and log back in to the terminal. The `(base)` show at the beginning of your command propmt.
8. Initailized conda to `.bash_profile` or `.zprofile`
  
   ````zsh
   conda init
   ````
   
10. Now remove the Anaconda installer with `rm <filename>`
11. Now create conda environment by command:

    ````zsh
    conda create --name <environment name> python=<version>
    ````
