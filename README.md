# llm-engine-service

## Build in Metal device

```shell
 cargo build --features metal --release
```


## Build in Cuda device

```shell
 cargo build --features cuda --release
```


### Prepare service on RunPod
#### prerequisite

```shell
apt update && apt install nano && apt-get install -y curl libssl-dev pkg-config
```

#### GitHub repo
```shell
ssh-keygen -t ed25519 -C "amin@thinksky.com"
```

path of keys : /root/.ssh/github

copy public key to GitHub ssh key setting
```shell
cat /root/.ssh/github
```

create a new ssh config file

```shell
nano /root/.ssh/config
```
copy these
```yaml
# Other github account: superman
Host github.com
  HostName github.com
  IdentityFile /root/.ssh/github
  IdentitiesOnly yes
```
then touch ssh config file

```shell
touch ~/.ssh/config
```

Fetch code on the machine
```shell
git clone git@github.com:aminnasiri/llm-engine-service.git
```

#### Install Rust
```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

touch Cargo config
```shell
. "$HOME/.cargo/env" 
```

### ENV set
```shell
export HF_TOKEN=xxx
```



