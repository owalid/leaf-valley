/**
 * Do NOT allow using `npm` as package manager.
 */
 if (process.env.npm_execpath.indexOf('yarn') === -1) {
  console.error('⛔️⛔️⛔️  You must use Yarn to install dependencies use: `$ yarn`  ⛔️⛔️⛔️');
  process.exit(1);
}