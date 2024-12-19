# pytrf

pytrf provides Python utilities for the analysis and combination of Terrestrial Reference Frames.
<br/><br/>

**Table of Contents**
1. [Installation](#installation)
    1. [Requirements](#requirements)
    1. [Clone the repository](#clone-the-repository)
    1. [Install pytrf](#install-pytrf)
    1. [Generate documentation](#generate-documentation)
    1. [Update with latest version](#update-with-latest-version)
1. [Tips and common issues](#tips-and-common-issues)  
    1. [Connectivity problems with `git`, `conda` and `pip`](#connectivity-problems-with-git-conda-and-pip)
        1. [IGN network](#ign-network)
        1. [Run `git` behind a proxy server](#run-git-behind-a-proxy-server)
        1. [Run `conda` behind a proxy server](#run-conda-behind-a-proxy-server)
        1. [Run `pip` behind a proxy server](#run-pip-behind-a-proxy-server)

<br/><br/>



## Installation

### Requirements

- `python3`. See [installation instructions](https://www.python.org/downloads/).
- `pip`. See [installation instructions](https://pip.pypa.io/en/stable/installation/).
- `git`. See [installation instructions](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

### Clone the repository

- Create a [github](https://github.com/) account if you don't already have one.
- Create a [**personal access token**](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token) for your github account. **Copy-paste your token value somewhere.**

- Open a terminal and **clone the repository** with the following command.  
  IMPORTANT: you will be asked to enter your github username and *password*. **The *password* is your token, not your usual github password**.
    ```sh
    $ git clone https://github.com/prebischung/pytrf
    ```

- A `pytrf` folder should have been created in the current directory.
  Run:
    ```sh
    $ cd pytrf
    ```

<br>

### Install pytrf

- If you work with a package manager, e.g., `conda`, you may want to create and activate a specific environment for pytrf.

- In `path_to_pytrf_folder/pytrf`, run one of the two following commands: use the `-e` option if you plan on coding within pytrf — you won't need to reinstall it after every change.

  Note: This will automatically install the missing required packages.
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
    You should hopefully be able to import the library without any error.

<br>

### Generate documentation

Run:
```sh
$ pdoc --html pytrf
```
A folder should have been created:
- `path_to_pytrf_folder/html`: contains the project documentation

<br>

### Update with latest version

```sh
$ git pull
```




<br></br>

## Tips and common issues

### Connectivity problems with `git`, `conda` and `pip`

`git`, `conda` and `pip` commands that require internet connection may fail because of the network configuration (firewall, proxy server, etc.)

If you do use a proxy and need to keep it activated for security reasons, this section will detail how to use `git`, `conda` and `pip` behind a proxy server.

Issues (and solutions) detailed were encountered (have worked) for a:
- Windows 11 machine
- connected to IGN network  

but can still be useful for users with a different configuration.

<br>

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

<br>

#### Run `git` behind a proxy server

##### Example of proxy error

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
>- Checking your git config file
>- Running `git config --global --get http.proxy`


Run the following command — replace `myproxy.com:portnumber` by the proxy address you have identified earlier):

```sh
git config --global http.proxy "myproxy.com:portnumber"
```

<br>


#### Run `conda` behind a proxy server

##### Example of proxy error

Issues may occur when trying to install libraries with conda:

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

- Check if you already have one created. Run:

    ```sh
    $ conda info
    ````

    And look for the field `user config file`.


    >**If `user config file` value is not null:**
    <br></br>
    >As a precaution, check if you already have a configuration value set for `proxy_servers.https` and copy paste the current value somewhere. You can get this value by:
    >- Checking the existing `.condarc` file
    >- Running the command `conda config --get proxy_servers.https`


- Run the following command — replace `myproxy.com:portnumber` by the proxy address you have identified earlier)

    ```sh
    $ conda config --set proxy_servers.https myproxy.com:portnumber
    ```

    If your `user config file` value was `None`, this command will create a `.condarc` in your home directory with the following content:

    ```txt
    proxy_servers:
      https: myproxy.com:portnumber
    ```

    If you already had a configuration file, it will simply add a row in it.

<br>

#### Run `pip` behind a proxy server

##### Example of proxy error

Issues may occur when trying to install libraries with `pip`:
```sh
(base) C:\Users\cpov>pip install library_name
WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ConnectTimeoutError(<pip._vendor.urllib3.connection.HTTPSConnection object at 0x000002395E6B2460>, 'Connection to pypi.org timed out. (connect timeout=15)')': /simple/library_name/
```

Like conda, one possibility is to modify the configuration file. As a temporary solution, we can indicate the proxy server address in the command, using the `proxy` option:

```sh
$ pip install --proxy myproxy.com:portnumber library_name
```
