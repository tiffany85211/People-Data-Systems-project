import random
import pickle
import os
import numpy as np

FILEPATH = 'agents/pricemaker_files/'

class Agent(object):
    def __init__(self, agent_number, params={}):
        self.this_agent_number = agent_number  # index for this agent
        self.opponent_number = 1 - agent_number  # index for opponent
        self.n_items = params["n_items"]

        # Unpickle the trained model
        # Complications: pickle should work with any machine learning models
        # However, this does not work with custom defined classes, due to the way pickle operates
        # TODO you can replace this with your own model
        self.trained_model = pickle.load(open(FILEPATH + 'logit_model', 'rb'))

        self.knn_model = pickle.load(open(FILEPATH + 'knn_model', 'rb'))
        self.user_embedding = pickle.load(open(FILEPATH + 'user_embedding', 'rb'))
        self.item_embedding = np.array([pickle.load(open(FILEPATH +'item0embedding', 'rb')), 
                                        pickle.load(open(FILEPATH + 'item1embedding', 'rb'))])

        self.alpha = 1.0

    def _process_last_sale(self, last_sale, profit_each_team):
        # print("last_sale: ", last_sale)
        # print("profit_each_team: ", profit_each_team)
        my_current_profit = profit_each_team[self.this_agent_number]
        opponent_current_profit = profit_each_team[self.opponent_number]

        my_last_prices = last_sale[2][self.this_agent_number]
        opponent_last_prices = last_sale[2][self.opponent_number]

        did_customer_buy_from_me = last_sale[1] == self.this_agent_number
        did_customer_buy_from_opponent = last_sale[1] == self.opponent_number

        which_item_customer_bought = last_sale[0]
        ratio = opponent_last_prices[which_item_customer_bought]/my_last_prices[which_item_customer_bought]
        if did_customer_buy_from_me:  # can increase prices
            self.alpha *= max(ratio * 0.95, 1.05)
        elif did_customer_buy_from_opponent:  # should decrease prices
            self.alpha *= max(min(ratio * 0.95, 0.95), 0.5)
        else:  # customer did not buy, should decrease prices even more so the customer buys
            self.alpha *= max(min(ratio * 0.8, 0.8), 0.5)

        # print("My current profit: ", my_current_profit)
        # print("Opponent current profit: ", opponent_current_profit)
        # print("My last prices: ", my_last_prices)
        # print("Opponent last prices: ", opponent_last_prices)
        # print("Did customer buy from me: ", did_customer_buy_from_me)
        # print("Did customer buy from opponent: ",
        #       did_customer_buy_from_opponent)
        # print("Which item customer bought: ", which_item_customer_bought)

        # TODO - add your code here to potentially update your pricing strategy based on what happened in the last round
        pass

    def _get_gradient_direction(self, min0, max0, min1, max1, buyer_vector):
        res_0, res_1, res_revenue = 0, 0, 0
        for p0 in [min0, max0]:
            for p1 in [min1, max1]:
                curr_revenue = self._get_pred_revenue(p0, p1, buyer_vector)
                if curr_revenue > res_revenue:
                    res_0, res_1, res_revenue = p0, p1, curr_revenue
        return res_0, res_1, res_revenue

    def _get_pred_revenue(self, p0, p1, buyer_vector):
        buyer_vector = np.append(buyer_vector, [p0, p1])
        pred = self.trained_model.predict_proba(buyer_vector.reshape(1, -1))
        return p0 * pred[0][1] + p1 * pred[0][2]

    def _get_optimal_price_without_competition(self, buyer_vector):
        max_iter = 100
        lr0, lr1 = 0.01, 0.02
        # initialize price
        price0, price1 = [1.1087160310662973, 2.028729190765244] # median from training_prices_decisions
        for _ in range(max_iter):
            curr_revenue = self._get_pred_revenue(price0, price1, buyer_vector)
            min0, max0 = price0 - lr0, price0 + lr0
            min1, max1 = price1 - lr1, price1 + lr1
            # use p0 +/- lr, and p1 +/- lr to calculate the combination {p0, p1} that gives the max revenue
            gradient_0, gradient_1, gradient_revenue = self._get_gradient_direction(min0, max0, min1, max1, buyer_vector)
            # if the max revenue from the new {p0, p1} combination is smaller than curr revenue -> curr price for item0 and item1 is maximum
            if gradient_revenue < curr_revenue:
                break
            price0, price1, curr_revenue = gradient_0, gradient_1, gradient_revenue
        return price0, price1, curr_revenue

    # Given an observation which is #info for new buyer, information for last iteration, and current profit from each time
    # Covariates of the current buyer, and potentially embedding. Embedding may be None
    # Data from last iteration (which item customer purchased, who purchased from, prices for each agent for each item (2x2, where rows are agents and columns are items)))
    # Returns an action: a list of length n_items=2, indicating prices this agent is posting for each item.
    def action(self, obs):
        new_buyer_covariates, new_buyer_embedding, last_sale, profit_each_team = obs

        # Estimate new_buyer_embedding if None
        if new_buyer_embedding is None:
            neighbor_ids = self.knn_model.kneighbors(new_buyer_covariates.reshape(1, -1), return_distance=False)[0]
            new_buyer_embedding = np.array(self.user_embedding.iloc[neighbor_ids].mean(axis=0))
        
        price0, price1, optimal_revenue = self._get_optimal_price_without_competition(np.concatenate([new_buyer_covariates, new_buyer_embedding @ self.item_embedding.T]))
        self._process_last_sale(last_sale, profit_each_team)

        return [price0 * self.alpha, price1 * self.alpha]
        # TODO Currently this output is just a deterministic 2-d array, but the students are expected to use the buyer covariates to make a better prediction
        # and to use the history of prices from each team in order to create prices for each item.
