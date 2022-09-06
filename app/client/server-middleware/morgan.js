// server-middleware/logging.js
import path from 'path';
import morgan from 'morgan';
const rfs = require('rotating-file-stream')

morgan.token('date', () => {
  return new Date().toISOString().replace("T", " ").split('.')[0] // get format YYYY-MM-DD hh:mm[:ss]
})
const accessLogStream = rfs.createStream('./access.log', {
  interval: '5d',
  path: path.join(__dirname, 'log')
})
export default morgan(':date[iso]',  { stream: accessLogStream });