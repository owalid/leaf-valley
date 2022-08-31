export const state = () => ({
  alert: null,
  currentTimeout: null,
})

export const mutations = {
  SET_ALERT(state, alert) {
    state.alert = alert
  },
}

export const actions = {
  ACTION_SET_ALERT({ commit }, alert) {
    commit('SET_ALERT', alert)
    state.currentTimeout = setTimeout(() => {
      commit('SET_ALERT', null)
    }, 5000)
  },
}

export const getters = {
  alert: (state) => state.alert,
}
