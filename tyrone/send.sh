#!/usr/bin/env bash
# run like `./send.sh` on local, to prepare remote to continue setup
# set up your ~/.ssh/config file to have an entry for 'mlx' to your Computa serfer

# like this:

# Host mlx
#  HostName <ip address>
#  IdentityFile ~/.ssh/<your saved private key file>
#  Port <port>
#  User root

if [[ -z "${1-}" ]]; then
    REMOTE="mlx"
else
    REMOTE="$1"
fi

# move private key, ssh.sh script and .env file to remote
scp ~/.ssh/id_ed25519 "$REMOTE:~/.ssh/id_ed25519"
scp ssh.sh "$REMOTE:ssh.sh"
scp .env "$REMOTE:.env"
