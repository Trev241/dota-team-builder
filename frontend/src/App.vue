<template>
  <div class="bg-slate-950 text-gray-200 font-commissioner">
    <nav
      v-if="showNavbar"
      class="flex text-white justify-between bg-gray-950 border-b border-gray-800 p-4"
    >
      <div>
        <h1 class="text-2xl font-semibold cursor-pointer" @click="$router.push('/')">
          DOTA Team Builder
        </h1>
      </div>
      <div class="flex items-center">
        <router-link to="/about" class="mr-6 hover:text-blue-300">About</router-link>
        <router-link to="/" class="mr-6 hover:text-blue-300">Builder</router-link>
        <router-link to="/embeddings" class="hover:text-blue-300">Synergies</router-link>
      </div>
    </nav>

    <main class="p-6">
      <router-view />
    </main>
  </div>
</template>

<script setup>
import { computed } from "vue"
import { useRoute } from "vue-router"
const route = useRoute()

const showNavbar = computed(() => !route.meta.hideNavbar)
</script>
