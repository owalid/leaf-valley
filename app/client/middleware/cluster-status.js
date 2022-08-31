// import axios from 'axios'

export default async function ({ redirect, $econome }) {
  if (process.env.NODE_ENV === 'production') {
    try {
      const { data } = await $econome.get(`/get-status`)
      if (data.State !== 'running') {
        redirect('/pending')
      }
    } catch (error) {
      redirect('/pending')
    }
  }
}
