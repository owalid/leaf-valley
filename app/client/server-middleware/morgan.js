// server-middleware/logging.js
import path from 'path';
import morgan from 'morgan';
const rfs = require('rotating-file-stream')

const accessLogStream = rfs.createStream('access.log', {
  interval: '5d',
  path: path.join(__dirname, 'log')
})
export default morgan(':date[iso]',  { stream: accessLogStream });