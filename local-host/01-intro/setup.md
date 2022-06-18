# ⚙ MacOS Apple Silicon | Environment Setup
---

## Anaconda:

**Download and install the Anaconda distribution.**
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
    
12. Then activate environment by the command:

    ````zsh
    conda activate <envinronment name>
    ````
    
    And deactivate envinronment by the command:
    
    ````zsh
    conda deactivate
    ````
    
    For check list of conda environment:
    
    ````zsh
    conda env list
    ````
 
13. Install others packages that are require.


## Docker:

For the `Docker` and `Docker-compose` in Apple Silicon architecture come together in the `Docker Desktop`. So we can download only one time.
Click &#8594; [Docker & Docker-compose](https://www.docker.com/products/docker-desktop/) and then follow the instructions to install it.

For run Docker without `sudo`:

Change your settings by run:

````zsh
sudo groupadd docker
````

````zsh
sudo gpasswd -a $USER docker
````

Then log out and log back in and run:

````zsh
sudo service docker restart
````

Test Docker can run successfully with:

````zsh
docker run hello-world
````

You should see something like this

![docker run hello-world](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/local-host/01-intro/images/Screen%20Shot%202565-06-18%20at%2022.23.29.png)
