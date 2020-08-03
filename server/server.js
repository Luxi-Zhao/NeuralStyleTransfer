const express = require('express')
const cors = require('cors')
const app = express()
const port = 8080
app.use(cors())

app.get('/', (req, res) => res.send('Hello World!'))

app.use(express.static('vgg19_model'))


app.listen(port, () => console.log(`Example app listening at http://localhost:${port}`))