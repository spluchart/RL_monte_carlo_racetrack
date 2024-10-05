import numpy as np
import matplotlib.pyplot as plt
import random


PLOT = True

COLOR_DICT = {
    't': (0, 0, 0), # track
    'g': (0, 255, 0), # grass
    's': (255, 0, 0), # starting line
    'f': (0, 0, 255), # finish line
    'c': (255, 255, 0) # car
    }



class Racetrack:
 
    def __init__(self, world_width=20, world_height=20, max_velocity=5):
        self.world_width = world_width
        self.world_height = world_height
        self.max_velocity = max_velocity
        
        self.state_space_dim = (
            self.world_width, # x
            self.world_height, # y
            self.max_velocity+1, # vx
            self.max_velocity+1 # vy
            )
            
        # lists all possible velocity increments (= all possible actions)
        # actions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
        self.actions = [(dvx, dvy) for dvx in [-1, 0, 1] for dvy in [-1, 0, 1]]       
        
        # lists for every velocity vector (vx, vy) tuple the indices of valid actions
        self.valid_actions = np.array([[self.list_valid_action_indices((vx, vy)) for vx in range(self.max_velocity+1)] for vy in range(self.max_velocity+1)], dtype=object)
        
        self.Q = np.full(self.state_space_dim+(len(self.actions),), -1000) # arbitrarily initialize the state values Q(s,a)
                
        self.C = np.zeros_like(self.Q) # cumulative sums of the weights initialized to 0
        
        # target policy, approximation of the optimal policy. Arbitrarily initialized.
        self.target_policy = np.full(self.state_space_dim, 4) 
                
        self.grid = self.grid_generation(self.world_width, self.world_height)
        self.finish_segment = self.finish_line_as_segment()
    
    # define a __repr__ that plots the (colored) grid and the target policy?

    def grid_generation(self, width, height):
        """ """
        grid = np.full((width, height), 'g') # grass element by default
        grid[0:5, :] = 't' # track
        grid[:, -5:] = 't' # track
        grid[0:5, 0] = 's' # starting line
        grid[-1,-5:] = 'f' # finish line
        # 'c' is for the car
        
        return grid
    
    
    def finish_line_as_segment(self):
        """ """
        x_list, y_list = np.where(self.grid == 'f')
        
        return [(x_list[0], y_list[0]), (x_list[-1], y_list[-1])]


    def grid_to_colormap(self, color_dict):
        """
        maps an array of token describing the world to color values usable by matplotlib 
        the mapping is described by color_dict
        color array has a (M, N, 3) shape to accommodate (R, G, B) values
        
        NOTE(Sylvain): color_array is the transpose of a grid array, to go from array orientation (row count from top to bottom) to (X, Y) coordinates on the plot
        """
        color_array = np.ndarray((self.grid.T.shape[0], self.grid.T.shape[1], 3))
        
        for k in color_dict:
            color_array[self.grid.T == k] = color_dict[k] 
        
        return color_array
    

    def list_valid_action_indices(self, vel):
        """ """
        indices = []
        
        for index, action in enumerate(self.actions):
            ## unpack components
            dvx, dvy = action
            vx, vy = vel
            
            ## projected speed at the next time step
            vx_next = vx + dvx
            vy_next = vy + dvy
            
            ## conditions on the velocity components
            forward = vx_next >= 0 and vy_next >= 0 and vx_next+vy_next > 0
            vel_limit = vx_next <= self.max_velocity and vy_next <= self.max_velocity
            
            is_valid = forward and vel_limit
        
            if is_valid:
                indices.append(index)
        
        return tuple(indices)
    
    
    def pick_random_position_on_token(self, token):
        """ """
        x_list, y_list = np.where(self.grid == token) # lists all cells containing the token
        index = np.random.choice(x_list.size) # picks randomly an index
        
        return x_list[index], y_list[index]
    
    
    def epsilon_greedy_policy(self, state, epsilon=0.2):
        """ """
        x, y, vx, vy = state
            
        ## lists the value function, for a given state and for all admissible actions
        action_indices = self.valid_actions[vx,vy] # admissible actions
        
        ## assign the greedy action with probability (1-epsilon)
        if np.random.random() > epsilon:
            prob = 1-epsilon
            q_list = np.take(self.Q[x, y, vx, vy, :], action_indices)
            index = random.choice(np.where(q_list == max(q_list))[0]) # picks the maximum value in q_list, breaks ties randomly
            action = action_indices[index]      
        else:
            prob = epsilon * (1.0/len(action_indices))         
            action = random.choice(action_indices)
  
        return action, prob
    
    def apply_target_policy(self, state):
        """ """
        action = self.target_policy[state]
        prob = 1.0
        
        return action, prob
    

    def generate_episode(self, starting_state, policy, max_steps=300):
        """ """
        
        ## initialization
        step = 0 # safeguard against infinite loops
        episode = []
        state = starting_state
                   
        ## loop while the car hasn't crossed the finish line
        while step < max_steps:       

            a, p = policy(state)
            episode.append((state,(a,p)))

            ## update the vehicle's velocity and position
            x, y, vx, vy = state # unpacks the state's description
            dvx, dvy = self.actions[a] # unpacks the velocity increments
            vx, vy = vx+dvx, vy+dvy
            x, y = position_update(x, y, vx, vy)
 
            ## test whether the vehicle crossed the finish line  
            s0 = [(episode[-1][0][0], episode[-1][0][1]), (x, y)] # last segment traveled by the vehicle 
            s1 = self.finish_segment
            
            if intersects(s0, s1):
                break
            
            ## test on the validity of the position
            # TODO(Sylvain): the test is done on the final position only,
            # we should rather make sure that the trajectory segment is not intersecting with non-track elements           
            try:
                out_of_track = self.grid[x, y] == 'g'
            except: # case where the position (x, y) is outside the grid
                out_of_track = True
            
            if out_of_track:
                vx, vy = 0, 0
                x, y = self.pick_random_position_on_token('s')
        
            state = (x, y, vx, vy) # update the state before taking the next action
            
            step += 1
            
        return episode
    
def position_update(x, y, vx, vy):
        """ """
        return x+vx, y+vy


def intersects(s0, s1):
    """
    assumes line segments are stored in the format [(x0,y0),(x1,y1)]
    
    seg1 = [[0, 0], [3, 3]]
    seg2 = [[20, 20], [3, 3.1]]
    print(intersects(seg1, seg2))
    """
    dx0 = s0[1][0]-s0[0][0]
    dx1 = s1[1][0]-s1[0][0]
    dy0 = s0[1][1]-s0[0][1]
    dy1 = s1[1][1]-s1[0][1]
    
    p0 = dy1*(s1[1][0]-s0[0][0]) - dx1*(s1[1][1]-s0[0][1])
    p1 = dy1*(s1[1][0]-s0[1][0]) - dx1*(s1[1][1]-s0[1][1])
    p2 = dy0*(s0[1][0]-s1[0][0]) - dx0*(s0[1][1]-s1[0][1])
    p3 = dy0*(s0[1][0]-s1[1][0]) - dx0*(s0[1][1]-s1[1][1])
    
    return (p0*p1<=0) & (p2*p3<=0)



## sandbox
test_case = Racetrack()


## off-policy MC control algorithm
max_episode = 30000
episode_count = 0

starting_state = (2,0,0,0)

gamma = 1

while episode_count < max_episode:

    print("==========", "episode #", episode_count, flush=True)
    
    episode = test_case.generate_episode(starting_state, test_case.epsilon_greedy_policy)
    
    G = 0
    W = 1
       
    for count, (state, (action_index,prob)) in enumerate(reversed(episode)):

        G = gamma*G + (-len(episode)+count)
        x, y, vx, vy = state
        test_case.C[x, y, vx, vy, action_index] += W
        test_case.Q[x, y, vx, vy, action_index] += (W/test_case.C[x, y, vx, vy, action_index])*(G-test_case.Q[x, y, vx, vy, action_index])
        
        test_case.target_policy[state] = test_case.epsilon_greedy_policy(state, epsilon=0.0)[0] # updates the target policy with the greedy action

        if action_index != test_case.target_policy[state]:
            break
        W = min(W/prob, 10000) # to avoid huge values
        

    episode_count += 1


episode = test_case.generate_episode(starting_state, test_case.apply_target_policy)
print(episode, flush=True)


if PLOT:

    for ((x,y,vx,vy), (action_index,prob)) in episode: # plots the vehicle's trajectory
        test_case.grid[x,y] = 'c'
    
    
    colormap = test_case.grid_to_colormap(COLOR_DICT)
    
    
    fig, axes = plt.subplots(figsize=(8,8))
    
    axes.grid(which='minor', linestyle='-', linewidth='0.3', color='white')  
    
    # Turns off the display of all ticks
    axes.tick_params(which='both', top=False, left=False, right=False, bottom=False, labelbottom=False, labelleft=False)
    axes.set_xticks(np.arange(0, test_case.world_width, 1), minor=True)
    axes.set_yticks(np.arange(0, test_case.world_height, 1), minor=True)
     
    plt.imshow(colormap, origin=[0,0], extent=[0,test_case.world_width,0,test_case.world_height])   
    plt.show()
