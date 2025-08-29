<template>
  <div class="max-w-6xl mx-auto mt-12">
    <h1 class="text-4xl mb-2 font-bold">ANOTHER DOTA SYNERGY PREDICTOR</h1>
    <p class="text-xl mb-6">A fun little tool to understand synergies with neural networks</p>

    <p class="text-gray-400 mb-4">
      If you're here, then I assume you must already know what Dota 2 is. For the uninformed, it's a
      simple game where two teams, each having five players, compete to destroy each other's
      "Ancient" (that's where the game gets its name from -
      <i>Defence of the Ancients</i>). Each player can choose a specific hero to play. To keep it
      very simple, it's like choosing a type of car to enter a race with. You can either choose a
      very lightweight car to speed across, or a monster truck to crush everyone's car in your way.
    </p>

    <p class="text-gray-400 mb-4">
      The goal is to predict the optimal team composition. A team that ideally consists of heroes
      forming the perfect synergy.
    </p>

    <h2 class="text-2xl font-bold mt-6 mb-2">How does it work?</h2>
    <p class="text-gray-400 mb-4">
      The neural network used here was trained on
      <a
        target="_blank"
        class="text-blue-400 hover:text-blue-300 hover:underline"
        href="https://archive.ics.uci.edu/dataset/367/dota2+games+results"
        >this dataset</a
      >
      containing about 90,000 matches. Each match contains information about the outcome of the
      match and the heroes that were played in it.
    </p>

    <p class="text-gray-400 mb-4">
      Every team has five heroes. For every match in the dataset, four out of five of the winning
      team's heroes were given to the model as input. It was the model's task to predict the fifth
      hero in the team from the initial four that were given. A task of this kind helps the model to
      create embeddings for each hero allowing it to capture latent relationships (such as
      synergies) between different heroes in the game.
    </p>

    <p class="text-gray-400 mb-4">
      Given how the model was trained, it will not be able to predict a team composition if less
      than four heroes are given as input. To overcome this, heroes for the second, third and fourth
      slots are recommended by measuring the cosine-similarity of different heroes in their
      embedding space.
    </p>

    <h2 class="text-2xl font-bold mt-6 mb-2">Where is Marci? Where is...?</h2>
    <p class="text-gray-400 mb-4">
      The
      <a
        target="_blank"
        class="text-blue-400 hover:text-blue-300 hover:underline"
        href="https://archive.ics.uci.edu/dataset/367/dota2+games+results"
        >dataset</a
      >
      used is from 2016. That's the reason why some heroes are missing. This was not created to be
      taken seriously which is why no effort has been made towards ensuring its modern relevance.
    </p>

    <h2 class="text-2xl font-bold mt-6 mb-2">Why can't I choose heroes for the enemy team?</h2>
    <p class="text-gray-400 mb-4">
      I suppose the biggest flaw with this model right now is that it does not factor in the
      composition of the opponent's team. The network was trained to understand and predict the
      synergies between heroes on the same team. A large part of the game also involves how heroes
      play <i>against</i> each other.
    </p>

    <p class="text-gray-400 mb-4">
      It's a shame that I never got around implementing this. Including counters and enemy picks
      increases the complexity of the model which means that it would require a significantly longer
      time to train. It's not impossible but considering that I had to rotate between three Google
      accounts to train the current model on Google Colab, I'd leave it for another day.
    </p>
  </div>
</template>
