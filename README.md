# Neural Style Transfer on the browser

This repo is a JavaScript version of Tensorflow's [Neural Style Transfer tutorial](https://www.tensorflow.org/tutorials/generative/style_transfer), developed based on the original style-transfer algorithm outlined in [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) (Gatys et al.)

## Demo Instance
Click [here](http://ec2-34-229-10-158.compute-1.amazonaws.com:3000/).

## Deployment
### Running the app on an EC2 instance
#### Setting up an EC2 instance
1. Create an EC2 instance with Amazon Linux 2 AMI 
2. SSH into the instance
3. Install docker and docker-compose
```
## Install docker
sudo yum install docker

## Start docker daemon
sudo systemctl start docker

## Install docker-compose
sudo curl -L https://github.com/docker/compose/releases/download/1.26.2/docker-compose-`uname -s`-`uname -m` -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

## Create the docker group 
sudo groupadd docker

## Add your user to the docker group
sudo usermod -aG docker ${USER}

## Verify that you can run docker commands without sudo
## (you may need to exit first then SSH again)
docker ps
```
4. Install git: `sudo yum install git`
5. Clone the project repo

#### Running the app
1.  `cd <project directory>/NeuralStyleTransfer`
2. Change the base URL in `docker-compose.yml` to the VM's public IP
3. Run `docker-compose up --build`
4. Navigate to `<vm public IP>:3000` on your browser

## Limitations
Your browser is going to be really slow when the app is running 😅.



