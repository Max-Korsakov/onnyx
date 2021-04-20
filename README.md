# FaceDemo
- Download and install [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) 
- Download and inslall Node version manager for Windows  [Node js](https://nodejs.org/en/) or google it for your system. If you do not find it for your system you have to install older version of Node js
- Download and inslall [Node js](https://nodejs.org/en/) LTS or older version from archive - 10.16.3 (in case you dont have nvm)
- Run in a command line 
```
node -v
npm -v
nvm -v // if you use nvm
```
to check versions. 

in case using nvm run
```
nvm install 10.16.3
```
```
nvm use 10.16.3
```
- navigate to the folder where you going to store project and run
```
git clone https://github.com/Max-Korsakov/onnyx
```
- run
```
npm i -g angular
```
- run
```
npm i @angular/cli
```
- navigate to the folder with project (cd ./onnyx) and run
```
npm i
```

copy model arcfaceresnet100-8_updated.onnx from the Mihail's zip file to the folder 'onnyx/face-demo/src/assets'

run

```
npm run start
```
go to the http://localhost:4200/


