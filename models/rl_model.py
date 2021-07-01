import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    def __init__(self, args):
        super(PolicyNetwork, self).__init__()
        self.args = args

        self.l1 = nn.Linear(args.state_dim, args.FF)
        self.l2 = nn.Linear(args.state_dim + args.action_dim, args.FF)
        self.l3 = nn.Linear(args.FF * 2, args.FF)
        self.l4 = nn.Linear(args.FF, args.action_dim)
        self.lstm1 = nn.LSTM(args.FF, args.FF)

        self.max_action = args.max_action
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state, last_action, hidden):
        self.lstm1.flatten_parameters()

        state = state.permute(1, 0, 2)
        last_action = last_action.permute(1, 0, 2)

        x = F.relu(self.l1(state))

        x_lstm = torch.cat([state, last_action], -1)
        x_lstm = F.relu(self.l2(x_lstm))
        x_lstm, x_lstm_hidden = self.lstm1(x_lstm, hidden) 

        x = torch.cat([x, x_lstm], -1)  
        x = F.relu(self.l3(x))
        x = self.l4(x)
        x = x.permute(1,0,2)

        output = x.reshape(-1, self.args.action_dim)
        return output, x_lstm_hidden

class QNetwork(nn.Module):
    def __init__(self, args):
        super(QNetwork, self).__init__()
        self.args = args

        self.l1 = nn.Linear(args.state_dim + args.action_dim, self.args.FF)
        self.l2 = nn.Linear(args.state_dim + args.action_dim, self.args.FF)
        self.l3 = nn.Linear(self.args.FF * 2, self.args.FF)
        self.l4 = nn.Linear(self.args.FF, 1)
        self.lstm1 = nn.LSTM(args.FF, args.FF)

    def forward(self, state, action, last_action, hidden):
        self.lstm1.flatten_parameters()

        state = state.permute(1, 0, 2)
        action = action.permute(1, 0, 2)
        last_action = last_action.permute(1, 0, 2)

        inputs = torch.cat([state, action], -1)
        x = F.relu(self.l1(inputs))

        x_lstm = torch.cat([state, last_action], -1)
        x_lstm = F.relu(self.l2(x_lstm))
        x_lstm, x_lstm_hidden = self.lstm1(x_lstm, hidden) 

        x = torch.cat([x, x_lstm], -1)  
        x = F.relu(self.l3(x))
        x = self.l4(x)
        x = x.permute(1,0,2)

        return x, x_lstm_hidden