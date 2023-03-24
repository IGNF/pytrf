# pytrf

pytrf provides Python utilities for the analysis and combination of Terrestrial Reference Frames.
<br/><br/>

**Table of Contents**
1. [Installation](#installation)
    1. [Install conda via Anaconda or Miniconda](#install-conda-via-anaconda-or-miniconda)
        1. [Clone the repository](#clone-the-repository)
        1. [Setup the conda environment](#setup-the-conda-environment)
        1. [Install pytrf](#install-pytrf)
        1. [Generate documentation](#generate-documentation)
        1. [Update with latest version](#update-with-latest-version)
1. [Tips and common issues](#tips-and-common-issues)  
    1. [IGN network](#ign-network)
    1. [Connectivity problems with `git`, `conda` and `pip`](#connectivity-problems-with-git,-conda-and-pip)
        1. [Run `git` behind a proxy server](#run-git-behind-a-proxy-server)
        1. [Run `conda` behind a proxy server](#run-conda-behind-a-proxy-server)
        1. ["Run `pip` behind a proxy server"](#run-pip-behind-a-proxy-server)
    1. [What is a virtual environment ?](#what-is-a-virtual-environment)
    1. [Conda useful commands](#conda-useful-commands)


<!--
  - [2. Clone the repository](#2-clone-the-repository)
  - [3. Setup the conda environment](#3-setup-the-conda-environment)
  - [4. Install pytrf](#4-install-pytrf')
  - [5. Generate documentation](5-generate-documentation)
  - [6. Update with latest version](6-update-with-latest-version)

 - [II. Tips and common issues](#II-tips-and-common-issues)
  - [1. Proxy issues with IGN network](1.-proxy-issues-with-ign-network)
      - [Connectivity problems with `git`, `conda` and `pip`](connectivity-problems-with-git,-conda-and-pip)
      - [IGN network](ign-network)
    - ["a. Run `git` behind a proxy server"](a.-run-`git`-behind-a-proxy-server)
    - ["b. Run `conda` behind a proxy server"](a.-run-`conda`-behind-a-proxy-server)
    - ["c. Run `pip` behind a proxy server"](c.-run-`pip`-behind-a-proxy-server)
  - [What is a virtual environment?](what-is-a-virtual-environment?)

-->
<br/><br/>



## I. Installation

The aim of this section is to setup a **conda virtual environment** with pytrf librairies and dependencies.  

This solution provides a first quick and minimal working environment for pytrf. Choose this option if you want to **test pytrf** tools for the first time, without the need of integrating it into another project at the moment.

> [**_What is a virtual environment?_**](#what-is-a-virtual-environment)


### Install conda via Anaconda or Miniconda

We will need `conda` (a package manager) for the installation. The easiest way to obtain it is to install an [**Anaconda**](https://docs.anaconda.com/anaconda/install/index.html) or [**Miniconda**](https://docs.conda.io/en/latest/miniconda.html) distribution.
If you already have one of those two distributions installed on your computer, please  directly go to the [next section](2.-clone-the-repository).

> [**_Anaconda or Miniconda?_**](https://conda.io/projects/conda/en/latest/user-guide/install/download.html#anaconda-or-miniconda)
<br/><br/>
> If one still can't figure out which distribution to install after reading this article, Miniconda installation is recommended for our purpose.


After installing the distribution, open a shell and run:
```sh
$ conda
```
If you have a error message "conda is not recognized as internal or external command.", it means that we have to set our PATH environment variable for Anaconda or Miniconda :

- First, check `conda.exe` installed location:
```sh
$ where conda.exe
```

  In the tested machine, `conda.exe` is in :
`C:\Users\cpov\Anaconda3\Scripts\`

- Add this directory to the PATH environment variable:
```sh
$ set Path=%Path%;C:\Users\cpov\Anaconda3\Scripts
```

  _Note: if you have installed Miniconda3, conda.exe may be located in :
`C:\Users\cpov\AppData\Local\miniconda3\Scripts`_

- Run
```sh
$ conda init
```

- For changes to take effect, close and re-open your current shell.

### Clone the repository

- Create a [github](https://github.com/) account if you don't already have one.
- Create a [**personal access token**](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token) for your github account. **Copy paste your token value somewhere.**

- Open a terminal and **clone the repository** with the following command.  
  IMPORTANT: you will be asked to enter your github username and *password*. **The *password* is your token, not your usual github password**.
```sh
$ git clone https://github.com/prebischung/pytrf
```

- A `pytrf` folder should now be seen in the current directory.  
  Run:
```sh
$ cd pytrf
```


### Setup the conda environment


~~We will use the `<path_to_pytrf_folder>/pytrf/environment.yml` file to set up our virtual environment.~~
Since this file is not in the repo at the moment, move it from your Downloads folder to `<path_to_pytrf_folder>/pytrf/`.

This formatted file lists all the packages required for pytrf (see above) and their dependencies.

| Packages | Purpose |
| ------ |------ |
| [python3]    | |
| [yaml]       | |
| [numpy]      | |
| [scipy]      | |
| [astropy]    | tools for astronomy and astrophysics|
| [matplotlib] | |
| [cartopy]    | geospatial data processing |
| [pip]        | |
| [git]        | |
| [pdoc3]      | Generate project documentation |

[python3]: https://www.python.org/downloads/
[yaml]: https://yaml.org/
[numpy]: https://numpy.org/
[scipy]: https://scipy.org/
[astropy]: https://www.astropy.org/
[matplotlib]: https://matplotlib.org/
[cartopy]: https://scitools.org.uk/cartopy/docs/latest/
[pip]: https://pypi.org/project/pip/
[git]: https://anaconda.org/anaconda/git
[pdoc3]: https://anaconda.org/anaconda/pdoc3


- Let's create a virtual environment named `pytrf_env` from the `environment.yml` file.
```sh
$ conda env create --name pytrf_env -f environment.yml
```

- Then activate it :
```sh
$ conda activate pytrf_env
```


### Install pytrf

- In `path_to_pytrf_folder/pytrf`, run one of the 2 following commands : use the `-e` option if you plan on coding within pytrf — you won't need to reinstall it after every change.

  Note: Don't miss the dot  **`.`** at the end of both commands.

  ```sh
  $ pip install .
  ```
  OR
  ```sh
  $ pip install -e .
  ```

- Open a python interpreter and try to import `pytrf`.
```sh
>>> import pytrf
>>>
>>> exit()
```
You should be able to import the library without any error.

### Generate documentation

```sh
$ pdoc --html pytrf
```
A folder should have been created:
- `html`: contains the project documentation


### Update with latest version

```sh
$ git pull
```




<br></br>

## Tips and common issues

### Proxy issues with IGN network

#### Connectivity problems with `git`, `conda` and `pip`


`git`, `conda` and `pip` commands that require internet connection may fail because of the network configuration (firewall, proxy server, etc.)

If you do use a proxy and need to keep it activated for security reasons, this section will detail how to use `git`, `conda` and `pip` behind a proxy server.


Issues (and solutions) detailed were encountered (has worked) for a :
- Windows 11 machine
- connected to IGN network  

but it can still be useful for users with a different configuration.


#### IGN network

It seems that IGN network configuration doesn't allow internet access when the proxy is deactivated.
In order to allow `git`, `conda` and `pip` to run connection-needed commands, we have to provide them our proxy address.

> How to get our proxy server address ?
<br></br>
> - In Windows search bar, go to: `Parameters` > `Internet and Network` > `Proxy`
<br></br>
> - In `Automatic proxy configuration`, `Use an installation script` should be activated. Copy the script address in your clipboard and paste it in a web browser.  
It should have download a `.pac` file in your usual `Downloads` folder.
<br></br>
> - Open the file. Identify your proxy address used for http requests (around the comment `// toutes les autres demandes (sauf ftp) => proxy`). The format is `myproxy.com:portnumber`.

<br></br>
##### Run `git` behind a proxy server

###### Example of proxy error

While running the `git clone` command, you may have the following error message.

```
(pytrf_env) C:\Users\cpov\pytrf>git clone https://github.com/prebischung/pytrf
Cloning into 'pytrf'...
fatal: unable to access 'https://github.com/prebischung/pytrf/': Failed to connect to github.com port 443 after 21091 ms: Timed out
```

This could be a proxy error.
There are several ways to solve this problem depending on your needs and actual github configuration file content.

One way is to set git config file (global or local one depending on your needs).

>As a precaution, and before making any changes, please check your `http.proxy` git variable first and copy paste the current value somewhere. You can get its value by :
- Checking your git config file
- Running `git config --global --get http.proxy`


Run the following command — replace `myproxy.com:portnumber` by the proxy address you have identified earlier):

```sh
git config --global http.proxy "myproxy.com:portnumber"
```

<br></br>

##### Run `conda` behind a proxy server


Issues may occur when trying to install librairies with conda:

```sh
(pytrf_env) C:\Users\cpov\pytrf> conda install pdoc3
Collecting package metadata (current_repodata.json): failed

CondaHTTPError: HTTP 000 CONNECTION FAILED for url <https://repo.anaconda.com/pkgs/main/win-64/current_repodata.json>
Elapsed: -

An HTTP error occurred when trying to retrieve this URL.
HTTP errors are often intermittent, and a simple retry will get you on your way.

If your current network has https://www.anaconda.com blocked, please file
a support request with your network engineering team.
```

The proxy address must be specified in conda configuration file (`.condarc`).

Check if you already have one created. Run:

```sh
$ conda info
````

And look for the field `user config file`.


>**If `user config file` value is not null:**
<br></br>
>As a precaution, check if you already have a configuration value set for `proxy_servers.https` and copy paste the current value somewhere. You can get this value by:
- Checking the existing `.condarc` file
- Running the command `conda config --get proxy_servers.https`


Run the following command — replace `myproxy.com:portnumber` by the proxy address you have identified earlier)

```sh
$ conda config --set proxy_servers.https myproxy.com:portnumber
```

If your `user config file` value was `None`, this command will create a `.condarc` in your home directory with the following content:

```txt
proxy_servers:
  https: myproxy.com:portnumber
```

If you already had a configuration file, it will simply add a row in it.

<br></br>

##### Run `pip` behind a proxy server
lorem ipsum

<br></br>

### What is a virtual environment?

**_[IN PROGRESS - NEED TO ADD SCHEMAS]_**

A virtual environment helps you keep dependencies required by different projects separated.

Why it is useful?

Let's say we work on a project1. To run it, we installed all the required dependencies in our machine :

We then start to work on another project (project2) which will never overlap with project1. It requires a more recent version of python3 and numpy.

To run both project1 AND project2 without conflicts, multiple options can be considered:
- upgrade python and numpy to run project2, and downgrade them each time project1 need to be run
- Install both versions manually and add them in python path with a different name.
- Test which python and numpy versions are compatible for both projects
- etc.

For a larger number of projects and packages, having to manage many different version of packages <b>in a single environment</b> can be quite time consuming and painful. In fact, with this approach, we just have <b>one global installation</b> for Python and librairies.

Virtual environment will allow us to have <b>separated environment for each project</b>, hence isolate dependencies : each project can have its own dependencies, regardless of what dependencies other projects have.

Other advantages of virtual environments are:
- easier to define and install packages specific to a project
- easier for other developers to reproduce your development environment.

### Conda useful commands

lorem ipsum
